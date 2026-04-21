# AudioToText

**Private, local speech-to-text for Windows (and other platforms that run Python).**  
Transcribe audio or video files—or record from your microphone—using **OpenAI Whisper** via [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper). Nothing is uploaded to the cloud; processing stays on your machine.

*(Optional: add a screenshot to this README with `![UI](docs/screenshot.png)` after you save an image under `docs/`.)*
<img width="1920" height="1045" alt="image" src="https://github.com/user-attachments/assets/46f407ab-bae1-46e1-a8ae-d555bd0ede75" />

---

## Features

| Feature | Description |
|--------|-------------|
| **Local & private** | Transcription runs offline after models are downloaded once. |
| **Whisper models** | Choose `tiny` → `large-v2` (accuracy vs speed trade-off). |
| **Languages** | Any language Whisper supports; **auto-detect** or pick a language. |
| **Files & mic** | Open media files (WAV, MP3, M4A, FLAC, MP4, MKV, …) or record. |
| **Optional split: Male / Female** | Heuristic **two-speaker** split using **MFCC clustering + pitch** (approximate; best with two clear voices). |
| **Compute** | **CPU only** or **Auto** (use **CUDA / GPU** when PyTorch sees one). |
| **Faster mode** | Optional **greedy** decoding (`beam_size=1`) for quicker runs with slightly lower quality. |
| **Timing** | Live **elapsed** clock plus a **breakdown** after a job (e.g. load · detection · model · transcribe). |
| **Media** | [FFmpeg](https://ffmpeg.org/) on `PATH` recommended; **imageio-ffmpeg** helps decode many formats without a system FFmpeg. |

---

## Requirements

- **Python** 3.10+ recommended  
- **RAM** — depends on model (e.g. **8 GB+** comfortable for `base`; more for `large`).  
- **GPU (optional)** — NVIDIA with CUDA for faster transcription when using **Auto**; otherwise CPU.  
- **FFmpeg** — optional but recommended for MP3/M4A/video containers (see app tip text).

---

## Installation

```bash
cd AudioToText
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python main.py
```

---

## Usage (quick)

1. **Model** — Start with `base` or `tiny` on slower PCs.  
2. **Spoken language** — Use **Auto-detect** or select the language.  
3. **Compute** — **Auto** if you have a supported NVIDIA GPU; else **CPU only**.  
4. **Split male / female** — Only if you need two columns; adds CPU work.  
5. **Faster transcription (greedy)** — Check for speed; uncheck for maximum default quality.  
6. **Browse** — Pick a file, or **Record**, then **Transcribe**.

**Copy / Save** — Exports text; split mode combines Male and Female sections in copy/save.

---

## Project layout

```
AudioToText/
├── main.py                 # Entry point
├── requirements.txt
└── audio_to_text/
    ├── app.py              # CustomTkinter UI, workers, timing
    ├── engine.py           # faster-whisper load & transcribe
    ├── voice_split.py      # MFCC + pitch split (optional)
    └── languages.py        # Language list for the UI
```

---

## How it works (short)

- **Transcription:** `faster-whisper` runs the Whisper model on **CPU** (`int8`) or **GPU** (`float16`) depending on settings.  
- **Split mode:** Audio is loaded at 16 kHz mono; windows are clustered; labels use **median pitch** (rough male/female mapping); each segment is transcribed separately.  
- **UI:** Background threads run inference; the main thread updates text via a queue (batched to avoid freezing the window).

---

## Limitations

- **Speaker split** is **heuristic** — not perfect diarization; labels can swap or collapse to one voice.  
- **Whisper** can hallucinate on silence or noisy audio—use **VAD** (on by default in the engine) and sensible models.  
- **Very old GPUs** may not be supported by current PyTorch/CUDA—you may effectively run **CPU-only**.

---

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper)  
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / [CTranslate2](https://github.com/OpenNMT/CTranslate2)  
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)  
- [librosa](https://librosa.org/), [scikit-learn](https://scikit-learn.org/)

---

## License

Specify your license here (e.g. MIT). Dependencies have their own licenses.

---

## GitHub “About” (one line)

**Local Whisper desktop app — private file/mic transcription, optional male/female split, GPU/CPU, greedy speed mode, per-stage timing.**
