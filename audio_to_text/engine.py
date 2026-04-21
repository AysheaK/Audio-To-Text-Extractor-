from __future__ import annotations

import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

SampleCallback = Callable[[np.ndarray], None]


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_transcribe_device(preference: str) -> str:
    """Map UI choice to a faster-whisper device string.

    - ``cpu`` — force CPU only.
    - ``auto`` — use CUDA GPU when available, otherwise CPU (Whisper uses one device at a time).
    """
    if preference == "cpu":
        return "cpu"
    return default_device()


def default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _transcribe_speed_params(fast: bool, beam_size: int = 5) -> tuple[int, bool]:
    """beam_size, condition_on_previous_text — fast mode trades a little quality for speed."""
    if fast:
        return 1, False
    return beam_size, True


class TranscriptionEngine:
    """Loads Whisper once per model key; runs transcription off the UI thread."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._key: tuple[str, str, str] | None = None
        self._lock = threading.Lock()

    def load_model(
        self,
        model_size: str,
        device: str | None = None,
        compute_type: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        device = device or default_device()
        compute_type = compute_type or default_compute_type(device)
        key = (model_size, device, compute_type)
        with self._lock:
            if self._model is not None and self._key == key:
                return
            if on_status:
                on_status(f"Loading model “{model_size}” ({device}, {compute_type})…")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._key = key
            if on_status:
                on_status("Model ready.")

    def transcribe_file(
        self,
        path: str | Path,
        language: str | None,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        fast: bool = False,
        on_segment: Callable[[str, float, float], None] | None = None,
        on_info: Callable[[str], None] | None = None,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        path = Path(path)
        bs, cond_prev = _transcribe_speed_params(fast, beam_size)
        segments, info = self._model.transcribe(
            str(path),
            language=language,
            beam_size=bs,
            vad_filter=vad_filter,
            condition_on_previous_text=cond_prev,
        )
        if on_info:
            lang = getattr(info, "language", None) or "?"
            prob = getattr(info, "language_probability", None)
            if prob is not None:
                on_info(f"Detected / using: {lang} ({prob:.0%} confidence)")
            else:
                on_info(f"Detected / using: {lang}")

        parts: list[str] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                parts.append(text)
                if on_segment:
                    on_segment(text, seg.start, seg.end)
        return "\n".join(parts).strip()

    def transcribe_buffer(
        self,
        audio: np.ndarray,
        samplerate: int,
        language: str | None,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        fast: bool = False,
        on_segment: Callable[[str, float, float], None] | None = None,
        on_info: Callable[[str], None] | None = None,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return transcribe_numpy(
            self._model,
            audio,
            samplerate,
            language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            fast=fast,
            on_segment=on_segment,
            on_info=on_info,
        )

    def transcribe_split_segments(
        self,
        wav: np.ndarray,
        sr: int,
        language: str | None,
        segments: list[tuple[float, float, str]],
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        fast: bool = False,
        on_partial: Callable[[str, str], None] | None = None,
        on_info: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Transcribe time ranges into male / female text buckets."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        male_parts: list[str] = []
        female_parts: list[str] = []
        segs = sorted(segments, key=lambda x: x[0])
        for t0, t1, gender in segs:
            i0 = max(0, int(t0 * sr))
            i1 = min(len(wav), int(t1 * sr))
            if i1 < i0 + int(0.1 * sr):
                continue
            chunk = wav[i0:i1]
            text = transcribe_numpy(
                self._model,
                chunk,
                sr,
                language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                fast=fast,
                on_segment=None,
                on_info=on_info,
            )
            text = (text or "").strip()
            if not text:
                continue
            if gender == "male":
                male_parts.append(text)
            else:
                female_parts.append(text)
            # Fewer UI updates + yield so the main thread can repaint (avoids "Not Responding").
            if on_partial and (len(male_parts) + len(female_parts)) % 2 == 0:
                on_partial("\n".join(male_parts).strip(), "\n".join(female_parts).strip())
            time.sleep(0.002)
        if on_partial and (male_parts or female_parts):
            total = len(male_parts) + len(female_parts)
            if total % 2 == 1:
                on_partial("\n".join(male_parts).strip(), "\n".join(female_parts).strip())
        return "\n".join(male_parts).strip(), "\n".join(female_parts).strip()


def save_wav_16k_mono(path: str | Path, samples: np.ndarray, samplerate: int) -> None:
    """Write 16-bit mono WAV. `samples` float32 [-1, 1] or int16."""
    path = Path(path)
    if samples.dtype != np.int16:
        samples = np.clip(samples.astype(np.float64), -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(samples.tobytes())


class MicRecorder:
    """Background mic capture to a float32 mono buffer at `samplerate`."""

    def __init__(self, samplerate: int = 16000) -> None:
        self.samplerate = samplerate
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def _callback(self, indata, _frames, _time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            pass
        block = indata.copy().reshape(-1)
        with self._lock:
            self._frames.append(block)

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._frames, axis=0)


def transcribe_numpy(
    model: "WhisperModel",
    audio: np.ndarray,
    samplerate: int,
    language: str | None,
    *,
    beam_size: int = 5,
    vad_filter: bool = True,
    fast: bool = False,
    on_segment: Callable[[str, float, float], None] | None = None,
    on_info: Callable[[str], None] | None = None,
) -> str:
    """Transcribe in-memory mono float32 audio (resampled to 16 kHz if needed)."""
    if audio.size < 1600:
        return ""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32)
    mx = float(np.max(np.abs(audio)))
    if mx > 1.0:
        audio = audio / mx
    if samplerate != 16000:
        ratio = 16000 / float(samplerate)
        x_old = np.arange(len(audio), dtype=np.float64)
        x_new = np.arange(0, len(audio) - 1, 1 / ratio)
        audio = np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)

    bs, cond_prev = _transcribe_speed_params(fast, beam_size)
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=bs,
        vad_filter=vad_filter,
        condition_on_previous_text=cond_prev,
    )
    if on_info:
        lang = getattr(info, "language", None) or "?"
        prob = getattr(info, "language_probability", None)
        if prob is not None:
            on_info(f"Detected / using: {lang} ({prob:.0%} confidence)")
        else:
            on_info(f"Detected / using: {lang}")
    parts: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
            if on_segment:
                on_segment(text, seg.start, seg.end)
    return "\n".join(parts).strip()
