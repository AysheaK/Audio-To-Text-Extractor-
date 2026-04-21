from __future__ import annotations

import queue
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np

from audio_to_text.engine import MicRecorder, TranscriptionEngine, default_device, resolve_transcribe_device
from audio_to_text.languages import sorted_language_choices
from audio_to_text.voice_split import load_audio_mono, split_voice_segments_by_gender

# —— Visual system (calm, high-contrast dark UI) ——
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

COLOR_BG = "#0c0c0f"
COLOR_CARD = "#14141a"
COLOR_CARD_BORDER = "#252530"
COLOR_ACCENT = "#818cf8"
COLOR_MUTED = "#8b92a6"
COLOR_OK = "#34d399"
COLOR_DANGER = "#f87171"

FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 22, "bold")
FONT_SUB = (FONT_FAMILY, 13)
FONT_BODY = (FONT_FAMILY, 13)
FONT_SMALL = (FONT_FAMILY, 11)

MODELS = ["tiny", "base", "small", "medium", "large-v2"]

# Display labels → engine.resolve_transcribe_device key
_COMPUTE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("CPU only", "cpu"),
    ("Auto (GPU if available)", "auto"),
)

# Drain at most this many queue events per timer tick so the UI thread keeps pumping
# Windows shows "Not Responding" if we process hundreds of text updates in one batch.
_POLL_MAX_EVENTS = 24


def _format_duration(seconds: float) -> str:
    """Human-readable duration for status / elapsed display."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m {s:.0f}s"


def _format_worker_error(exc: BaseException) -> str:
    """Some exceptions stringify to '' — Tk messagebox would show a blank body."""
    name = type(exc).__name__
    msg = str(exc).strip()
    if msg:
        return f"{name}: {msg}"
    return (
        f"{name} (no message text). "
        "Try: install FFmpeg and add it to PATH for video/MP3/M4A, use a WAV file, "
        "or turn off “Split male / female voices” if analysis fails."
    )


def _card(master: ctk.CTkBaseClass, **kwargs: object) -> ctk.CTkFrame:
    return ctk.CTkFrame(
        master,
        fg_color=COLOR_CARD,
        corner_radius=14,
        border_width=1,
        border_color=COLOR_CARD_BORDER,
        **kwargs,
    )


class AudioToTextApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AudioToText")
        self.minsize(1920, 1080)
        self.geometry("1000x700")
        self.configure(fg_color=COLOR_BG)

        self._engine = TranscriptionEngine()
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._recorder: MicRecorder | None = None
        self._job_start: float | None = None
        self._elapsed_after_id: str | None = None

        self._build_ui()
        self.after(80, self._poll_queue)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        side = _card(self, width=300)
        side.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="AudioToText", font=FONT_TITLE, text_color="white").pack(anchor="w", padx=20, pady=(20, 4))
        ctk.CTkLabel(
            side,
            text="Local speech → text. Any language Whisper supports.\nRuns on your machine (private).",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
            justify="left",
            wraplength=250,
        ).pack(anchor="w", padx=20, pady=(0, 16))

        ctk.CTkLabel(side, text="Model", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).pack(anchor="w", padx=20)
        self._model = ctk.CTkOptionMenu(side, values=MODELS, font=FONT_BODY, height=36)
        self._model.set("base")
        self._model.pack(fill="x", padx=20, pady=(6, 14))

        ctk.CTkLabel(side, text="Spoken language", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).pack(anchor="w", padx=20)
        self._language_choices = sorted_language_choices()
        labels = [c[1] for c in self._language_choices]
        self._lang = ctk.CTkOptionMenu(side, values=labels, font=FONT_BODY, height=36)
        self._lang.set(labels[0])
        self._lang.pack(fill="x", padx=20, pady=(6, 14))

        self._split_mode = tk.BooleanVar(value=False)
        self._split_cb = ctk.CTkCheckBox(
            side,
            text="Split male / female voices",
            variable=self._split_mode,
            command=self._apply_split_layout,
            font=FONT_BODY,
        )
        self._split_cb.pack(anchor="w", padx=20, pady=(0, 8))
        ctk.CTkLabel(
            side,
            text="Uses MFCC clustering + pitch (approximate). Best with two clear speakers.",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
            wraplength=250,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 14))

        ctk.CTkLabel(side, text="Compute", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).pack(
            anchor="w", padx=20
        )
        compute_labels = [x[0] for x in _COMPUTE_OPTIONS]
        self._compute_menu = ctk.CTkOptionMenu(
            side,
            values=compute_labels,
            font=FONT_BODY,
            height=36,
            command=self._on_compute_change,
        )
        self._compute_menu.set(compute_labels[1] if default_device() == "cuda" else compute_labels[0])
        self._compute_menu.pack(fill="x", padx=20, pady=(6, 4))
        self._compute_hint = ctk.CTkLabel(
            side,
            text="",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
            wraplength=250,
            justify="left",
        )
        self._compute_hint.pack(anchor="w", padx=20, pady=(0, 8))
        self._update_compute_hint()

        self._fast_mode = tk.BooleanVar(value=False)
        self._fast_cb = ctk.CTkCheckBox(
            side,
            text="Faster transcription (greedy)",
            variable=self._fast_mode,
            font=FONT_BODY,
        )
        self._fast_cb.pack(anchor="w", padx=20, pady=(0, 4))
        ctk.CTkLabel(
            side,
            text="Uses beam size 1 and less context — quicker, slightly less accurate.",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
            wraplength=250,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 14))

        ctk.CTkLabel(
            side,
            text="Tip: Install FFmpeg and add it to PATH for MP3/M4A/Video.",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
            wraplength=250,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(8, 20))

        theme_row = ctk.CTkFrame(side, fg_color="transparent")
        theme_row.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(theme_row, text="Appearance", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).pack(side="left")
        self._theme = ctk.CTkSegmentedButton(theme_row, values=["Dark", "Light"], command=self._on_theme)
        self._theme.set("Dark")
        self._theme.pack(side="right")

        # Main workspace
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        main.grid_rowconfigure(2, weight=1)
        main.grid_columnconfigure(0, weight=1)

        hero = ctk.CTkFrame(main, fg_color="transparent")
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(hero, text="Turn audio into text", font=(FONT_FAMILY, 20, "bold"), text_color="white").pack(anchor="w")
        ctk.CTkLabel(
            hero,
            text="Choose a file or record from your microphone. Transcription stays on this computer.",
            font=FONT_SUB,
            text_color=COLOR_MUTED,
        ).pack(anchor="w", pady=(4, 0))

        file_card = _card(main)
        file_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        file_card.grid_columnconfigure(1, weight=1)

        self._path_var = tk.StringVar(value="")
        ctk.CTkLabel(file_card, text="Audio file", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(14, 6)
        )
        entry = ctk.CTkEntry(
            file_card,
            textvariable=self._path_var,
            placeholder_text="Supported: WAV, MP3, M4A, FLAC, MP4, MKV… (FFmpeg)",
            height=40,
            font=FONT_BODY,
            border_color=COLOR_CARD_BORDER,
        )
        entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(16, 8), pady=(0, 14))
        file_card.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(
            file_card,
            text="Browse…",
            width=110,
            height=40,
            font=FONT_BODY,
            fg_color=COLOR_ACCENT,
            hover_color="#6d73e8",
            command=self._browse,
        ).grid(row=1, column=2, padx=(0, 16), pady=(0, 14))

        out_card = _card(main)
        out_card.grid(row=2, column=0, sticky="nsew")
        out_card.grid_rowconfigure(1, weight=1)
        out_card.grid_columnconfigure(0, weight=1)
        out_card.grid_columnconfigure(1, weight=1)

        self._hdr_unified = ctk.CTkLabel(
            out_card,
            text="Transcript",
            font=(FONT_FAMILY, 12, "bold"),
            text_color=COLOR_MUTED,
        )
        self._hdr_unified.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 8))

        self._hdr_split = ctk.CTkFrame(out_card, fg_color="transparent")
        self._hdr_split.grid_columnconfigure(0, weight=1)
        self._hdr_split.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self._hdr_split, text="Male", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 8)
        )
        ctk.CTkLabel(self._hdr_split, text="Female", font=(FONT_FAMILY, 12, "bold"), text_color=COLOR_MUTED).grid(
            row=0, column=1, sticky="w", padx=16, pady=(14, 8)
        )
        self._hdr_split.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._hdr_split.grid_remove()

        tb_kw = dict(
            font=(FONT_FAMILY, 14),
            wrap="word",
            fg_color="#0f0f14",
            border_color=COLOR_CARD_BORDER,
            border_width=1,
            corner_radius=10,
        )
        self._text = ctk.CTkTextbox(out_card, **tb_kw)
        self._text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=16, pady=(0, 12))
        self._text.insert("1.0", "Transcription will appear here.\n\n")
        self._text.configure(state="disabled")

        self._split_pane = ctk.CTkFrame(out_card, fg_color="transparent")
        self._split_pane.grid_rowconfigure(0, weight=1)
        self._split_pane.grid_columnconfigure(0, weight=1)
        self._split_pane.grid_columnconfigure(1, weight=1)
        self._text_male = ctk.CTkTextbox(self._split_pane, **tb_kw)
        self._text_female = ctk.CTkTextbox(self._split_pane, **tb_kw)
        self._text_male.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(0, 12))
        self._text_female.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=(0, 12))
        self._text_male.insert("1.0", "Male voice text will appear here.\n")
        self._text_female.insert("1.0", "Female voice text will appear here.\n")
        self._text_male.configure(state="disabled")
        self._text_female.configure(state="disabled")
        self._split_pane.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        self._split_pane.grid_remove()

        actions = ctk.CTkFrame(out_card, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 14))
        actions.grid_columnconfigure(6, weight=1)

        self._btn_transcribe = ctk.CTkButton(
            actions,
            text="Transcribe file",
            width=140,
            height=40,
            font=(FONT_FAMILY, 13, "bold"),
            fg_color=COLOR_ACCENT,
            hover_color="#6d73e8",
            command=self._on_transcribe_file,
        )
        self._btn_transcribe.grid(row=0, column=0, padx=4)

        self._btn_record = ctk.CTkButton(
            actions,
            text="● Record",
            width=120,
            height=40,
            font=FONT_BODY,
            fg_color="#334155",
            hover_color="#475569",
            command=self._toggle_record,
        )
        self._btn_record.grid(row=0, column=1, padx=4)

        self._btn_copy = ctk.CTkButton(
            actions,
            text="Copy",
            width=88,
            height=40,
            font=FONT_BODY,
            command=self._copy,
        )
        self._btn_copy.grid(row=0, column=2, padx=4)

        self._btn_save = ctk.CTkButton(
            actions,
            text="Save…",
            width=88,
            height=40,
            font=FONT_BODY,
            command=self._save,
        )
        self._btn_save.grid(row=0, column=3, padx=4)

        self._btn_clear = ctk.CTkButton(
            actions,
            text="Clear",
            width=88,
            height=40,
            font=FONT_BODY,
            fg_color="transparent",
            border_width=1,
            border_color=COLOR_CARD_BORDER,
            command=self._clear_text,
        )
        self._btn_clear.grid(row=0, column=4, padx=4)

        self._elapsed_label = ctk.CTkLabel(
            actions,
            text="",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
            width=130,
            anchor="e",
        )
        self._elapsed_label.grid(row=0, column=5, sticky="e", padx=(4, 4))

        self._status = ctk.CTkLabel(actions, text="Ready", font=FONT_SMALL, text_color=COLOR_MUTED)
        self._status.grid(row=0, column=6, sticky="e", padx=8)

        self._progress = ctk.CTkProgressBar(main, mode="indeterminate", height=6, progress_color=COLOR_ACCENT)
        self._progress.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self._progress.grid_remove()

    def _apply_split_layout(self) -> None:
        split = self._split_mode.get()
        if split:
            self._hdr_unified.grid_remove()
            self._hdr_split.grid(row=0, column=0, columnspan=2, sticky="ew")
            self._text.grid_remove()
            self._split_pane.grid(row=1, column=0, columnspan=2, sticky="nsew")
        else:
            self._hdr_split.grid_remove()
            self._hdr_unified.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 8))
            self._split_pane.grid_remove()
            self._text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=16, pady=(0, 12))

    def _on_theme(self, value: str) -> None:
        ctk.set_appearance_mode("dark" if value == "Dark" else "light")

    def _selected_language(self) -> str | None:
        label = self._lang.get()
        for code, name in self._language_choices:
            if name == label:
                return code
        return None

    def _compute_preference(self) -> str:
        label = self._compute_menu.get()
        for display, key in _COMPUTE_OPTIONS:
            if display == label:
                return key
        return "auto"

    def _on_compute_change(self, _value: str | None = None) -> None:
        self._update_compute_hint()

    def _update_compute_hint(self) -> None:
        resolved = resolve_transcribe_device(self._compute_preference())
        if resolved == "cuda":
            self._compute_hint.configure(
                text="Whisper uses: GPU (CUDA). Speaker detection (MFCC) uses CPU.",
                text_color=COLOR_OK,
            )
        else:
            self._compute_hint.configure(
                text="Whisper uses: CPU. Pick “Auto” to use GPU when CUDA is available.",
                text_color=COLOR_MUTED,
            )

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio or video",
            filetypes=[
                ("Media", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.mp4 *.mkv *.webm *.mov *.avi"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)

    def _stop_elapsed_timer(self) -> None:
        if self._elapsed_after_id is not None:
            try:
                self.after_cancel(self._elapsed_after_id)
            except Exception:
                pass
            self._elapsed_after_id = None
        self._job_start = None
        self._elapsed_label.configure(text="")

    def _schedule_elapsed_tick(self) -> None:
        if not self._busy or self._job_start is None:
            return
        elapsed = time.monotonic() - self._job_start
        self._elapsed_label.configure(text=f"Elapsed: {_format_duration(elapsed)}")
        self._elapsed_after_id = self.after(500, self._schedule_elapsed_tick)

    def _set_busy(self, busy: bool, *, show_progress: bool = False) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self._btn_transcribe.configure(state=state)
        self._btn_record.configure(state=state)
        self._btn_copy.configure(state=state)
        self._btn_save.configure(state=state)
        self._split_cb.configure(state=state)
        self._compute_menu.configure(state=state)
        self._fast_cb.configure(state=state)
        if busy:
            self._stop_elapsed_timer()
            self._job_start = time.monotonic()
            self._elapsed_label.configure(text=f"Elapsed: {_format_duration(0)}")
            self._schedule_elapsed_tick()
        else:
            self._stop_elapsed_timer()
        if busy and show_progress:
            self._progress.grid()
            self._progress.start()
        else:
            self._progress.stop()
            self._progress.grid_remove()

    def _write_text(self, content: str, *, append: bool = False) -> None:
        self._text.configure(state="normal")
        if not append:
            self._text.delete("1.0", "end")
        self._text.insert("end", content)
        self._text.configure(state="disabled")
        self._text.see("end")

    def _write_split_text(self, male: str, female: str) -> None:
        for tb, s in (
            (self._text_male, male),
            (self._text_female, female),
        ):
            tb.configure(state="normal")
            tb.delete("1.0", "end")
            tb.insert("1.0", s.strip() or "(No speech detected)")
            tb.configure(state="disabled")
            tb.see("end")

    def _clear_text(self) -> None:
        if self._split_mode.get():
            self._write_split_text("", "")
        else:
            self._write_text("")

    def _copy(self) -> None:
        if self._split_mode.get():
            self._text_male.configure(state="normal")
            self._text_female.configure(state="normal")
            m = self._text_male.get("1.0", "end").strip()
            f = self._text_female.get("1.0", "end").strip()
            self._text_male.configure(state="disabled")
            self._text_female.configure(state="disabled")
            t = f"Male\n{m}\n\nFemale\n{f}".strip()
        else:
            self._text.configure(state="normal")
            t = self._text.get("1.0", "end").strip()
            self._text.configure(state="disabled")
        if t:
            self.clipboard_clear()
            self.clipboard_append(t)
            self._set_status("Copied to clipboard")
        else:
            self._set_status("Nothing to copy")

    def _save(self) -> None:
        if self._split_mode.get():
            self._text_male.configure(state="normal")
            self._text_female.configure(state="normal")
            m = self._text_male.get("1.0", "end").strip()
            f = self._text_female.get("1.0", "end").strip()
            self._text_male.configure(state="disabled")
            self._text_female.configure(state="disabled")
            t = f"Male\n{m}\n\nFemale\n{f}".strip()
        else:
            self._text.configure(state="normal")
            t = self._text.get("1.0", "end").strip()
            self._text.configure(state="disabled")
        if not t:
            messagebox.showinfo("Save", "No text to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if path:
            Path(path).write_text(t, encoding="utf-8")
            self._set_status(f"Saved: {path}")

    def _on_transcribe_file(self) -> None:
        if self._busy:
            return
        path = self._path_var.get().strip()
        if not path or not Path(path).is_file():
            messagebox.showwarning("Audio file", "Please choose a valid audio or video file.")
            return
        lang = self._selected_language()
        model = self._model.get()
        transcribe_device = resolve_transcribe_device(self._compute_preference())
        use_fast = self._fast_mode.get()

        if self._split_mode.get():

            def worker_split() -> None:
                try:
                    t0 = time.monotonic()
                    self._queue.put(("status", "Loading audio…"))
                    wav, sr = load_audio_mono(path)
                    t_load = time.monotonic() - t0
                    segments, note = split_voice_segments_by_gender(
                        wav,
                        sr,
                        on_status=lambda s: self._queue.put(("status", s)),
                    )
                    t_detect = time.monotonic() - t0 - t_load
                    self._queue.put(("status", note))
                    self._queue.put(("status", "Loading model…"))
                    self._engine.load_model(
                        model,
                        device=transcribe_device,
                        on_status=lambda s: self._queue.put(("status", s)),
                    )
                    t_model = time.monotonic() - t0 - t_load - t_detect
                    self._queue.put(("status", "Transcribing by speaker…"))

                    def on_partial(male: str, female: str) -> None:
                        self._queue.put(("partial_split", (male, female)))

                    def on_info(line: str) -> None:
                        self._queue.put(("status", line))

                    male_t, female_t = self._engine.transcribe_split_segments(
                        wav,
                        sr,
                        lang,
                        segments,
                        fast=use_fast,
                        on_partial=on_partial,
                        on_info=on_info,
                    )
                    t_transcribe = time.monotonic() - t0 - t_load - t_detect - t_model
                    total = time.monotonic() - t0
                    timing = (
                        f"total {_format_duration(total)} "
                        f"(load {_format_duration(t_load)} · "
                        f"detection {_format_duration(t_detect)} · "
                        f"model {_format_duration(t_model)} · "
                        f"transcribe {_format_duration(t_transcribe)})"
                    )
                    self._queue.put(("done_split", (male_t, female_t, timing)))
                except Exception as e:
                    traceback.print_exc()
                    self._queue.put(("error", _format_worker_error(e)))

            self._set_busy(True, show_progress=True)
            self._set_status("Working…")
            threading.Thread(target=worker_split, daemon=True).start()
            return

        def worker() -> None:
            try:
                t0 = time.monotonic()
                self._queue.put(("status", "Loading model…"))
                self._engine.load_model(
                    model,
                    device=transcribe_device,
                    on_status=lambda s: self._queue.put(("status", s)),
                )
                t_model = time.monotonic() - t0
                self._queue.put(("status", "Transcribing…"))
                full: list[str] = []

                def on_seg(text: str, _s: float, _e: float) -> None:
                    full.append(text)
                    self._queue.put(("partial", "\n".join(full)))

                def on_info(line: str) -> None:
                    self._queue.put(("status", line))

                text = self._engine.transcribe_file(
                    path,
                    lang,
                    fast=use_fast,
                    on_segment=on_seg,
                    on_info=on_info,
                )
                t_transcribe = time.monotonic() - t0 - t_model
                total = time.monotonic() - t0
                timing = (
                    f"total {_format_duration(total)} "
                    f"(model {_format_duration(t_model)} · transcribe {_format_duration(t_transcribe)})"
                )
                self._queue.put(("done", (text, timing)))
            except Exception as e:
                traceback.print_exc()
                self._queue.put(("error", _format_worker_error(e)))

        self._set_busy(True, show_progress=True)
        self._set_status("Working…")
        threading.Thread(target=worker, daemon=True).start()

    def _toggle_record(self) -> None:
        if self._busy:
            return
        if self._recorder is None:
            self._recorder = MicRecorder(samplerate=16000)
            try:
                self._recorder.start()
            except Exception as e:
                self._recorder = None
                messagebox.showerror("Microphone", f"Could not start recording:\n{e}")
                return
            self._btn_record.configure(text="■ Stop", fg_color=COLOR_DANGER, hover_color="#dc2626")
            self._set_status("Recording… speak now")
            self._btn_transcribe.configure(state="disabled")
            return

        audio = self._recorder.stop()
        self._recorder = None
        self._btn_record.configure(text="● Record", fg_color="#334155", hover_color="#475569")
        self._btn_transcribe.configure(state="normal")

        if audio.size < 3200:
            self._set_status("Recording too short — try again")
            return

        lang = self._selected_language()
        model = self._model.get()
        transcribe_device = resolve_transcribe_device(self._compute_preference())
        use_fast = self._fast_mode.get()

        if self._split_mode.get():

            def worker_rec_split() -> None:
                try:
                    t0 = time.monotonic()
                    wav = audio.astype(np.float32)
                    segments, note = split_voice_segments_by_gender(
                        wav,
                        16000,
                        on_status=lambda s: self._queue.put(("status", s)),
                    )
                    t_detect = time.monotonic() - t0
                    self._queue.put(("status", note))
                    self._queue.put(("status", "Loading model…"))
                    self._engine.load_model(
                        model,
                        device=transcribe_device,
                        on_status=lambda s: self._queue.put(("status", s)),
                    )
                    t_model = time.monotonic() - t0 - t_detect
                    self._queue.put(("status", "Transcribing recording by speaker…"))

                    def on_partial(male: str, female: str) -> None:
                        self._queue.put(("partial_split", (male, female)))

                    def on_info(line: str) -> None:
                        self._queue.put(("status", line))

                    male_t, female_t = self._engine.transcribe_split_segments(
                        wav,
                        16000,
                        lang,
                        segments,
                        fast=use_fast,
                        on_partial=on_partial,
                        on_info=on_info,
                    )
                    t_transcribe = time.monotonic() - t0 - t_detect - t_model
                    total = time.monotonic() - t0
                    timing = (
                        f"total {_format_duration(total)} "
                        f"(detection {_format_duration(t_detect)} · "
                        f"model {_format_duration(t_model)} · "
                        f"transcribe {_format_duration(t_transcribe)})"
                    )
                    self._queue.put(("done_split", (male_t, female_t, timing)))
                except Exception as e:
                    traceback.print_exc()
                    self._queue.put(("error", _format_worker_error(e)))

            self._set_busy(True, show_progress=True)
            threading.Thread(target=worker_rec_split, daemon=True).start()
            return

        def worker() -> None:
            try:
                t0 = time.monotonic()
                self._queue.put(("status", "Loading model…"))
                self._engine.load_model(
                    model,
                    device=transcribe_device,
                    on_status=lambda s: self._queue.put(("status", s)),
                )
                t_model = time.monotonic() - t0
                self._queue.put(("status", "Transcribing recording…"))

                def on_seg(text: str, _s: float, _e: float) -> None:
                    pass

                def on_info(line: str) -> None:
                    self._queue.put(("status", line))

                text = self._engine.transcribe_buffer(
                    audio,
                    16000,
                    lang,
                    fast=use_fast,
                    on_segment=on_seg,
                    on_info=on_info,
                )
                t_transcribe = time.monotonic() - t0 - t_model
                total = time.monotonic() - t0
                timing = (
                    f"total {_format_duration(total)} "
                    f"(model {_format_duration(t_model)} · transcribe {_format_duration(t_transcribe)})"
                )
                self._queue.put(("done", (text, timing)))
            except Exception as e:
                traceback.print_exc()
                self._queue.put(("error", _format_worker_error(e)))

        self._set_busy(True, show_progress=True)
        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        n = 0
        try:
            while n < _POLL_MAX_EVENTS:
                kind, payload = self._queue.get_nowait()
                n += 1
                if kind == "status":
                    self._set_status(str(payload))
                elif kind == "partial":
                    self._write_text(str(payload), append=False)
                elif kind == "partial_split":
                    male, female = payload  # type: ignore[misc]
                    self._write_split_text(str(male), str(female))
                elif kind == "done":
                    self._set_busy(False, show_progress=False)
                    if isinstance(payload, tuple) and len(payload) == 2:
                        text, timing = str(payload[0]), str(payload[1])
                    else:
                        text, timing = str(payload), ""
                    self._write_text(text.strip() or "(No speech detected)", append=False)
                    self._set_status(f"Done — {timing}" if timing else "Done")
                elif kind == "done_split":
                    self._set_busy(False, show_progress=False)
                    pl = payload  # type: ignore[misc]
                    if isinstance(pl, tuple) and len(pl) == 3:
                        male, female, timing = str(pl[0]), str(pl[1]), str(pl[2])
                    else:
                        male, female, timing = str(pl[0]), str(pl[1]), ""
                    self._write_split_text(male, female)
                    self._set_status(f"Done — {timing}" if timing else "Done")
                elif kind == "error":
                    self._set_busy(False, show_progress=False)
                    err_text = str(payload).strip() or _format_worker_error(
                        RuntimeError("Unknown error (empty message from worker)")
                    )
                    messagebox.showerror("AudioToText", err_text)
                    self._set_status("Error")
        except queue.Empty:
            pass
        # If we hit the cap, more events are waiting — poll again soon so the window stays responsive.
        delay_ms = 1 if n >= _POLL_MAX_EVENTS else 80
        self.after(delay_ms, self._poll_queue)


def run() -> None:
    app = AudioToTextApp()
    app.mainloop()


if __name__ == "__main__":
    run()
