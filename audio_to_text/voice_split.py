"""Split overlapping voices into male/female buckets using MFCC clustering + pitch.

This is a lightweight heuristic (no GPU, no HuggingFace tokens). It clusters short
time windows into two speakers, labels clusters by median F0 (lower → male), then
hands time ranges to Whisper for transcription.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import librosa

OnStatus = Callable[[str], None]


def _needs_ffmpeg_fallback(exc: Exception) -> bool:
    """librosa/audioread raises this when no system FFmpeg is available."""
    if type(exc).__name__ == "NoBackendError":
        return True
    msg = str(exc).lower()
    return "nobackend" in msg or "no backend" in msg


def _load_pcm_via_imageio_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode with the FFmpeg binary shipped by imageio-ffmpeg (no PATH setup)."""
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError(
            "No FFmpeg on your system PATH. Install FFmpeg, or run: pip install imageio-ffmpeg"
        ) from e
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()[:1200]
        raise RuntimeError(f"ffmpeg decode failed: {err or proc.returncode}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise RuntimeError("ffmpeg produced no audio samples.")
    wav = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return wav, target_sr


def load_audio_mono(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    p = Path(path)
    try:
        wav, sr = librosa.load(str(p), sr=target_sr, mono=True)
    except Exception as e:
        if _needs_ffmpeg_fallback(e):
            try:
                wav, sr = _load_pcm_via_imageio_ffmpeg(p, target_sr)
            except Exception as e2:
                d1 = str(e).strip() or type(e).__name__
                d2 = str(e2).strip() or type(e2).__name__
                raise RuntimeError(
                    f"Could not load audio from {p.name!r}. Librosa: ({d1}). Fallback: ({d2}). "
                    "Install FFmpeg to PATH or: pip install imageio-ffmpeg"
                ) from e2
        else:
            detail = str(e).strip() or type(e).__name__
            raise RuntimeError(
                f"Could not load audio from {p.name!r} ({detail}). "
                "For MP4/MKV/M4A/MP3, install FFmpeg and add it to your system PATH, "
                "or convert the file to WAV."
            ) from e
    if wav.size == 0:
        raise RuntimeError(f"Loaded audio from {p.name!r} but it is empty (0 samples).")
    return wav.astype(np.float32), int(sr)


def _rms_energy(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk))))


def median_f0_hz(wav: np.ndarray, sr: int) -> float:
    if wav.size < 2048:
        return float("nan")
    f0, _, _ = librosa.pyin(
        y=wav,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=160,
    )
    f0 = f0[~np.isnan(f0)]
    if f0.size == 0:
        return float("nan")
    return float(np.median(f0))


def _mfcc_window_features(
    wav: np.ndarray,
    sr: int,
    *,
    window_sec: float = 1.0,
    hop_sec: float = 0.25,
    hop_length: int = 160,
    energy_floor: float = 0.012,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Returns feature vectors and (start_sample, end_sample) per window."""
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    n_frames = mfcc.shape[1]
    if n_frames < 4:
        return [], [], []

    frame_to_sec = hop_length / float(sr)
    win_frames = max(1, int(round(window_sec / frame_to_sec)))
    hop_frames = max(1, int(round(hop_sec / frame_to_sec)))

    feats: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []

    for start_f in range(0, n_frames - win_frames + 1, hop_frames):
        end_f = start_f + win_frames
        m = mfcc[:, start_f:end_f]
        d = delta[:, start_f:end_f]
        feat = np.concatenate([np.mean(m, axis=1), np.mean(d, axis=1)])

        start_s = int(start_f * hop_length)
        end_s = min(int(end_f * hop_length), len(wav))
        chunk = wav[start_s:end_s]
        if _rms_energy(chunk) < energy_floor:
            continue

        feats.append(feat)
        spans.append((start_s, end_s))

    return feats, spans


def _merge_same_label(
    spans: list[tuple[int, int]],
    labels: list[int],
    sr: int,
) -> list[tuple[float, float, int]]:
    """Merge overlapping time windows that share the same cluster label."""
    if not spans:
        return []
    out: list[tuple[float, float, int]] = []
    cs, ce = spans[0]
    cl = labels[0]
    for i in range(1, len(spans)):
        s, e = spans[i]
        lab = labels[i]
        if lab == cl:
            ce = max(ce, e)
        else:
            out.append((cs / sr, ce / sr, cl))
            cs, ce, cl = s, e, lab
    out.append((cs / sr, ce / sr, cl))
    return out


def split_voice_segments_by_gender(
    wav: np.ndarray,
    sr: int,
    *,
    on_status: OnStatus | None = None,
) -> tuple[list[tuple[float, float, str]], str]:
    """Cluster audio into two voices; label clusters male/female from median pitch.

    Returns (segments, note) where each segment is (start_sec, end_sec, "male"|"female").
    """
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = wav.astype(np.float32)
    if wav.size < int(2.5 * sr):
        dur = wav.size / float(sr)
        return [(0.0, dur, "male")], "Short clip — treating as a single voice."

    def st(msg: str) -> None:
        if on_status:
            on_status(msg)

    st("Analyzing speakers (MFCC clustering)…")
    feats, spans = _mfcc_window_features(wav, sr)
    if len(feats) < 6:
        dur = wav.size / float(sr)
        return [(0.0, dur, "male")], "Not enough voiced windows — single transcript."

    x = np.stack(feats, axis=0)
    x = StandardScaler().fit_transform(x)

    sil = 0.0
    if len(x) >= 8:
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(x)
        try:
            sil = float(silhouette_score(x, labels))
        except Exception:
            sil = 0.0
    else:
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(x)

    if sil < 0.06 and len(x) >= 8:
        dur = wav.size / float(sr)
        return [(0.0, dur, "male")], "Speakers not well separated — single transcript."

    merged = _merge_same_label(spans, list(labels), sr)
    merged = [m for m in merged if m[1] - m[0] >= 0.12]

    if not merged:
        dur = wav.size / float(sr)
        return [(0.0, dur, "male")], "No segments after merge — single transcript."

    st("Estimating pitch per speaker…")
    f0_by_cluster: dict[int, list[float]] = {0: [], 1: []}
    for t0, t1, cid in merged:
        i0 = int(t0 * sr)
        i1 = int(t1 * sr)
        if i1 <= i0:
            continue
        f0 = median_f0_hz(wav[i0:i1], sr)
        if not np.isnan(f0):
            f0_by_cluster.setdefault(int(cid), []).append(f0)

    v0 = f0_by_cluster.get(0, [])
    v1 = f0_by_cluster.get(1, [])
    med0 = float(np.median(v0)) if v0 else float("nan")
    med1 = float(np.median(v1)) if v1 else float("nan")

    if not np.isnan(med0) and not np.isnan(med1):
        if med0 <= med1:
            cmap = {0: "male", 1: "female"}
        else:
            cmap = {0: "female", 1: "male"}
        note = f"Clusters mapped by pitch (approx. {med0:.0f} Hz vs {med1:.0f} Hz)."
    else:
        cmap = {0: "male", 1: "female"}
        note = "Pitch data thin — using default cluster order (male/female may swap)."

    st("Speaker regions ready.")
    out: list[tuple[float, float, str]] = []
    for t0, t1, cid in merged:
        g = cmap.get(int(cid), "male")
        out.append((float(t0), float(t1), g))

    return out, note
