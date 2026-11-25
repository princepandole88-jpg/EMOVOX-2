import io
import wave
from face_emotion import FaceEmotionMonitor
import os
import re
import sys
import json
import time
import types
import queue
import asyncio
import threading
from datetime import datetime
from tkinter import filedialog
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import edge_tts
import pygame
import customtkinter as ctk
from dotenv import load_dotenv
from groq import Groq
from mem0 import Memory
from ddgs import DDGS

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

MOOD_PROSODY = {
    "stressed":   ("-6%", "-2Hz", "#e67e22", 0.38),
    "anxious":    ("-6%", "-2Hz", "#e67e22", 0.38),
    "sad":        ("-8%", "-3Hz", "#5dade2", 0.35),
    "tired":      ("-8%", "-3Hz", "#85929e", 0.35),
    "frustrated": ("-4%", "-1Hz", "#ec7063", 0.42),
    "happy":      ("+6%", "+2Hz", "#58d68d", 0.55),
    "excited":    ("+8%", "+3Hz", "#2ecc71", 0.60),
    "curious":    ("+2%", "+1Hz", "#af7ac5", 0.48),
    "neutral":    ("+0%", "+0Hz", "#3498db", 0.45),
}

RAW_SAMPLE_PATH = "raw_voice_sample.wav"
CLEAN_VOICE_PATH = "cloned_indian_voice.wav"

class _ChatBoxShim:
    def __init__(self, app):
        self._app = app
        self._buf = ""

    def configure(self, **kwargs):
        pass

    def delete(self, *args):
        self._app._clear_bubbles()

    def insert(self, index, text, *tags):
        self._buf += str(text)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                sender, msg = line.split(":", 1)
                self._app.append_chat(sender.strip(), msg.strip())
            else:
                self._app.append_chat("System", line)

    def see(self, *args):
        pass

    def yview_moveto(self, *args):
        pass

EMOVOX_IDENTITY_GUARDRAIL = (
    "STRICT IDENTITY, UNIVERSAL TRUTH & ANTI-SYCOPHANCY RULES:\n"
    "1. IDENTITY: Your name is ALWAYS EMOVOX, an empathetic desktop AI companion created by Pushpendra Pandole and Puneet Kushwaha. Never pretend to be an 'Emu', animal, or human.\n"
    "2. UNIVERSAL TRUTHS & FACTS: Never agree with factually false claims (e.g., wrong math, false science, false history, or false statements about your identity) even if the user insists, argues, or says 'no, you are wrong'.\n"
    "3. RESPECTFUL FIRMNESS: When a user contradicts a universal truth or verified fact, acknowledge their message warmly without being rude, clearly restate the true fact, and stand firm on reality.\n"
    "4. EMOTIONAL EMPATHY WITHOUT LYING: Validate the user's feelings (e.g., frustration or curiosity), NEVER validate false facts.\n"
    "5. CONCISE STYLE: Keep replies natural, friendly, and 1-3 sentences long.\n"
)

class EmovoxApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EMOVOX — Emotional AI Companion")
        self.geometry("900x670")
        self.minsize(760, 560)

        self.groq_client = None
        self.model_name = None
        self.memory = None
        self.user_id = "user_primary"

        self.tts_model = None
        self.voice_ready = False
        self.tts_lock = threading.Lock()
        self.voice_mode = ctk.StringVar(value="Female (Neerja)")
        self.detected_expression = "neutral"
        try:
            self._log_mood_history("neutral")
        except Exception:
            pass
        self.user_is_sad = False
        self.face_cam_enabled = True
        self.face_monitor = None

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.fs = 16000

        self._build_ui()
        self.update()
        threading.Thread(target=self._init_backend, daemon=True).start()
        self._start_face_monitor()

    def _get_theme(self):
        if getattr(self, "theme_mode", "light") == "dark":
            return {
                "outer_bg": "#2B0812",
                "title_fg": "#FDF2F5",
                "seg_bg": "#1B040B",
                "seg_sel": "#6E152C",
                "seg_sel_hover": "#851A36",
                "seg_unsel_hover": "#420B1A",
                "seg_fg": "#FDF2F5",
                "theme_btn_bg": "#1B040B",
                "theme_btn_hover": "#420B1A",
                "theme_btn_fg": "#F5D0DA",
                "cam_on_bg": "#6E152C",
                "cam_on_hover": "#851A36",
                "cam_on_fg": "#FDF2F5",
                "cam_off_bg": "#1B040B",
                "cam_off_hover": "#420B1A",
                "cam_off_fg": "#B88A97",
                "clear_hover": "#420B1A",
                "clear_fg": "#E6B8C4",
                "pill_bg": "#1B040B",
                "pill_fg": "#F5D0DA",
                "chat_mood_bg": "#22060E",
                "chat_mood_fg": "#E6B8C4",
                "sheet_bg": "#14060A",
                "sheet_border": "#471120",
                "scroll_btn": "#471120",
                "scroll_hover": "#6E152C",
                "input_bg": "#1C080E",
                "input_border": "#471120",
                "entry_fg": "#FDF2F5",
                "entry_ph": "#9E6B79",
                "send_bg": "#7A1830",
                "send_hover": "#941E3B",
                "send_fg": "#FFFFFF",
                "mic_bg": "#2B0B14",
                "mic_hover": "#3D101D",
                "mic_fg": "#F5D0DA",
                "status_fg": "#D9A7B5",
                "user_bubble": "#6E152C",
                "user_fg": "#FFFFFF",
                "user_ts": "#F3D0D8",
                "bot_bubble": "#210910",
                "bot_border": "#471120",
                "bot_sender": "#E89DB0",
                "bot_fg": "#F7E6EB",
                "sys_fg": "#A87483",
            }
        return {
            "outer_bg": "#4A0D1D",
            "title_fg": "#FFFFFF",
            "seg_bg": "#340713",
            "seg_sel": "#7A1830",
            "seg_sel_hover": "#8F1D39",
            "seg_unsel_hover": "#5E1226",
            "seg_fg": "#FFFFFF",
            "theme_btn_bg": "#340713",
            "theme_btn_hover": "#5E1226",
            "theme_btn_fg": "#F7DCE2",
            "cam_on_bg": "#F4ECEE",
            "cam_on_hover": "#E8DCE0",
            "cam_on_fg": "#5C1124",
            "cam_off_bg": "#340713",
            "cam_off_hover": "#5E1226",
            "cam_off_fg": "#E5C3CC",
            "clear_hover": "#5E1226",
            "clear_fg": "#F7DCE2",
            "pill_bg": "#F4ECEE",
            "pill_fg": "#5C1124",
            "chat_mood_bg": "#EFE5E8",
            "chat_mood_fg": "#5C1124",
            "sheet_bg": "#F4ECEE",
            "sheet_border": "#DCC8CE",
            "scroll_btn": "#DFC4CB",
            "scroll_hover": "#C79FA9",
            "input_bg": "#F4ECEE",
            "input_border": "#DCC8CE",
            "entry_fg": "#231116",
            "entry_ph": "#9E737E",
            "send_bg": "#5C1124",
            "send_hover": "#450B1A",
            "send_fg": "#FFFFFF",
            "mic_bg": "#EAE0E3",
            "mic_hover": "#DFD0D5",
            "mic_fg": "#5C1124",
            "status_fg": "#F5D5DD",
            "user_bubble": "#5C1124",
            "user_fg": "#FFFFFF",
            "user_ts": "#F3D0D8",
            "bot_bubble": "#EBE0E4",
            "bot_border": "#DCC8CE",
            "bot_sender": "#6B142A",
            "bot_fg": "#231116",
            "sys_fg": "#8F5D6A",
        }

    def _toggle_theme(self):
        self.theme_mode = "dark" if getattr(self, "theme_mode", "light") == "light" else "light"
        ctk.set_appearance_mode(self.theme_mode)
        t = self._get_theme()

        self.configure(fg_color=t["outer_bg"])
        self.title_lbl.configure(text_color=t["title_fg"])
        self.theme_btn.configure(
            text="Light" if self.theme_mode == "dark" else "Dark",
            fg_color=t["theme_btn_bg"],
            hover_color=t["theme_btn_hover"],
            text_color=t["theme_btn_fg"],
        )
        self.voice_switch.configure(
            fg_color=t["seg_bg"],
            selected_color=t["seg_sel"],
            selected_hover_color=t["seg_sel_hover"],
            unselected_color=t["seg_bg"],
            unselected_hover_color=t["seg_unsel_hover"],
            text_color=t["seg_fg"],
        )
        if getattr(self, "face_cam_enabled", True):
            self.cam_toggle_btn.configure(
                fg_color=t["cam_on_bg"],
                hover_color=t["cam_on_hover"],
                text_color=t["cam_on_fg"],
            )
        else:
            self.cam_toggle_btn.configure(
                fg_color=t["cam_off_bg"],
                hover_color=t["cam_off_hover"],
                text_color=t["cam_off_fg"],
            )
        self.clear_btn.configure(hover_color=t["clear_hover"], text_color=t["clear_fg"])

        self.chat_mood_badge.configure(fg_color=t["chat_mood_bg"], text_color=t["chat_mood_fg"])
        self._on_face_update(
            getattr(self, "detected_expression", "neutral") != "no_face",
            getattr(self, "detected_expression", "neutral"),
            getattr(self, "user_is_sad", False),
            0.0,
        )

        self.chat_sheet.configure(fg_color=t["sheet_bg"], border_color=t["sheet_border"])
        self.chat_scroll.configure(
            fg_color=t["sheet_bg"],
            scrollbar_button_color=t["scroll_btn"],
            scrollbar_button_hover_color=t["scroll_hover"],
        )

        self.input_pill.configure(fg_color=t["input_bg"], border_color=t["input_border"])
        self.msg_entry.configure(text_color=t["entry_fg"], placeholder_text_color=t["entry_ph"])
        self.send_btn.configure(fg_color=t["send_bg"], hover_color=t["send_hover"], text_color=t["send_fg"])
        if hasattr(self, "mic_btn"):
            self.mic_btn.configure(fg_color=t["mic_bg"], hover_color=t["mic_hover"], text_color=t["mic_fg"])
        self.status_label.configure(text_color=t["status_fg"])

        for entry in getattr(self, "_bubble_refs", []):
            kind = entry[0]
            if kind == "user":
                _, bubble, msg_lbl, ts_lbl = entry
                if bubble.winfo_exists():
                    bubble.configure(fg_color=t["user_bubble"])
                    msg_lbl.configure(text_color=t["user_fg"])
                    ts_lbl.configure(text_color=t["user_ts"])
            elif kind == "bot":
                _, bubble, hdr_lbl, msg_lbl = entry
                if bubble.winfo_exists():
                    bubble.configure(fg_color=t["bot_bubble"], border_color=t["bot_border"])
                    hdr_lbl.configure(text_color=t["bot_sender"])
                    msg_lbl.configure(text_color=t["bot_fg"])
            elif kind == "system":
                _, sys_lbl = entry
                if sys_lbl.winfo_exists():
                    sys_lbl.configure(text_color=t["sys_fg"])

    def _capture_mic_audio(self):
        import speech_recognition as sr
        r = sr.Recognizer()
        r.energy_threshold = 260
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.75

        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.35)
                audio = r.listen(source, timeout=7, phrase_time_limit=10)
            return r, audio
        except Exception as pyaudio_err:
            try:
                import sounddevice as sd
                fs = 16000
                dur = 5.0
                rec = sd.rec(int(dur * fs), samplerate=fs, channels=1, dtype="int16")
                sd.wait()
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(rec.tobytes())
                audio = sr.AudioData(buf.getvalue()[44:], fs, 2)
                return r, audio
            except Exception as sd_err:
                raise RuntimeError(f"Mic backend unavailable (PyAudio: {pyaudio_err} | sounddevice: {sd_err})")

    def _on_mic_click(self):
        if getattr(self, "_mic_listening", False):
            self._mic_listening = False
            try:
                t = self._get_theme()
                self.mic_btn.configure(text="Mic", fg_color=t["mic_bg"], text_color=t["mic_fg"])
                self.status_label.configure(text="Ready")
            except Exception:
                pass
            return

        self._mic_listening = True
        try:
            self.mic_btn.configure(text="Stop", fg_color="#E11D48", text_color="#FFFFFF")
            self.status_label.configure(text="Listening... Speak now")
        except Exception:
            pass

        def _reset_ui(status_msg="Ready"):
            self._mic_listening = False
            try:
                t = self._get_theme()
                self.mic_btn.configure(text="Mic", fg_color=t["mic_bg"], text_color=t["mic_fg"])
                self.status_label.configure(text=status_msg)
            except Exception:
                pass

        def _submit_spoken(text):
            _reset_ui("Ready")
            try:
                self.msg_entry.delete(0, "end")
                self.msg_entry.insert(0, text)
                if hasattr(self, "send_btn"):
                    self.send_btn.invoke()
            except Exception as send_err:
                _reset_ui(f"Send error: {send_err}")

        def _worker():
            try:
                r, audio = self._capture_mic_audio()
                if not getattr(self, "_mic_listening", False):
                    return
                self.after(0, lambda: self.status_label.configure(text="Transcribing..."))
                spoken = r.recognize_google(audio)
                if spoken and spoken.strip():
                    self.after(0, lambda s=spoken.strip(): _submit_spoken(s))
                else:
                    self.after(0, lambda: _reset_ui("No speech detected — try again"))
            except Exception as e:
                err_type = type(e).__name__
                if "WaitTimeoutError" in err_type:
                    msg = "Mic timed out — no voice heard"
                elif "UnknownValueError" in err_type:
                    msg = "Could not understand audio — speak clearly"
                elif "RequestError" in err_type:
                    msg = "Speech service offline — check internet"
                else:
                    msg = f"Mic error: {e}"
                self.after(0, lambda m=msg: _reset_ui(m))

        threading.Thread(target=_worker, daemon=True).start()

    def _log_mood_history(self, emotion, source="webcam"):
        emo = str(emotion or "").strip().lower()
        if not emo or emo in ("none", "unknown", "detecting...", "off"):
            return
        now = time.time()
        last_emo = getattr(self, "_last_logged_mood", None)
        last_ts = getattr(self, "_last_logged_mood_ts", 0.0)
        if emo == last_emo and (now - last_ts) < 60.0:
            return
        self._last_logged_mood = emo
        self._last_logged_mood_ts = now

        hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mood_history.json")
        try:
            records = []
            if os.path.exists(hist_path):
                with open(hist_path, "r", encoding="utf-8") as hf:
                    records = json.load(hf)
                    if not isinstance(records, list):
                        records = []
            records.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "emotion": emo,
                "source": source,
            })
            records = records[-1000:]
            with open(hist_path, "w", encoding="utf-8") as hf:
                json.dump(records, hf, indent=2)
        except Exception:
            pass

    def _get_mood_trend_summary(self):
        hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mood_history.json")
        if not os.path.exists(hist_path):
            return "No mood history recorded yet. Turn on the camera or chat with me to start tracking!"
        try:
            with open(hist_path, "r", encoding="utf-8") as hf:
                records = json.load(hf)
            if not records:
                return "Mood history log is currently empty."
            counts = {}
            for r in records:
                e = str(r.get("emotion", "neutral")).capitalize()
                counts[e] = counts.get(e, 0) + 1
            total = len(records)
            ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            breakdown = ", ".join(f"{k}: {v} ({int(round(v * 100 / total))}%)" for k, v in ranked)
            latest = records[-1]
            return (
                f"Long-Term Emotional Trend ({total} check-ins logged):\n"
                f"- Breakdown: {breakdown}\n"
                f"- Dominant Mood: {ranked[0][0]}\n"
                f"- Latest Reading: {str(latest.get('emotion', '')).capitalize()} at {latest.get('timestamp', '')}"
            )
        except Exception as e:
            return f"Could not read mood history: {e}"

    def _speak_offline_pyttsx3(self, text):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 172)
            voices = eng.getProperty("voices") or []
            for v in voices:
                if "zira" in v.name.lower() or "female" in v.name.lower():
                    eng.setProperty("voice", v.id)
                    break
            eng.say(str(text))
            eng.runAndWait()
            return True
        except Exception:
            return False

    def _build_ui(self):
        if not hasattr(self, "theme_mode"):
            self.theme_mode = "light"
        self._bubble_refs = []
        t = self._get_theme()

        self.title("EMOVOX")
        self.geometry("880x700")
        self.minsize(720, 540)
        self.configure(fg_color=t["outer_bg"])

        topbar = ctk.CTkFrame(self, height=56, fg_color="transparent")
        topbar.pack(fill="x", padx=28, pady=(16, 4))
        topbar.pack_propagate(False)

        self.title_lbl = ctk.CTkLabel(
            topbar,
            text="EMOVOX",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=t["title_fg"],
        )
        self.title_lbl.pack(side="left")

        right_controls = ctk.CTkFrame(topbar, fg_color="transparent")
        right_controls.pack(side="right")

        if not hasattr(self, "voice_mode"):
            self.voice_mode = ctk.StringVar(value="Female (Neerja)")

        self.voice_switch = ctk.CTkSegmentedButton(
            right_controls,
            values=["Female (Neerja)", "Female (Swara)"],
            variable=self.voice_mode,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=t["seg_bg"],
            selected_color=t["seg_sel"],
            selected_hover_color=t["seg_sel_hover"],
            unselected_color=t["seg_bg"],
            unselected_hover_color=t["seg_unsel_hover"],
            text_color=t["seg_fg"],
            corner_radius=10,
            height=29,
        )
        self.voice_switch.pack(side="left", padx=(0, 8))

        self.theme_btn = ctk.CTkButton(
            right_controls,
            text="Dark" if self.theme_mode == "light" else "Light",
            width=58,
            height=29,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=t["theme_btn_bg"],
            hover_color=t["theme_btn_hover"],
            text_color=t["theme_btn_fg"],
            command=self._toggle_theme,
        )
        self.theme_btn.pack(side="left", padx=(0, 6))

        self.cam_toggle_btn = ctk.CTkButton(
            right_controls,
            text="Cam: ON",
            width=74,
            height=29,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=t["cam_on_bg"],
            hover_color=t["cam_on_hover"],
            text_color=t["cam_on_fg"],
            command=self._toggle_face_cam,
        )
        self.cam_toggle_btn.pack(side="left", padx=(0, 6))

        self.clear_btn = ctk.CTkButton(
            right_controls,
            text="Clear",
            width=54,
            height=29,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="transparent",
            hover_color=t["clear_hover"],
            text_color=t["clear_fg"],
            command=self._clear_bubbles,
        )
        self.clear_btn.pack(side="left")

        mood_strip = ctk.CTkFrame(self, height=34, fg_color="transparent")
        mood_strip.pack(fill="x", padx=28, pady=(0, 8))
        mood_strip.pack_propagate(False)

        self.webcam_mood_badge = ctk.CTkLabel(
            mood_strip,
            text="WEBCAM MOOD: NEUTRAL",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=t["pill_bg"],
            text_color=t["pill_fg"],
            corner_radius=12,
            padx=12,
            pady=3,
        )
        self.webcam_mood_badge.pack(side="left", padx=(0, 8))

        self.chat_mood_badge = ctk.CTkLabel(
            mood_strip,
            text="CHAT MOOD: NEUTRAL",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=t["chat_mood_bg"],
            text_color=t["chat_mood_fg"],
            corner_radius=12,
            padx=12,
            pady=3,
        )
        self.chat_mood_badge.pack(side="left")

        self.chat_sheet = ctk.CTkFrame(
            self,
            fg_color=t["sheet_bg"],
            corner_radius=20,
            border_width=1,
            border_color=t["sheet_border"],
        )
        self.chat_sheet.pack(fill="both", expand=True, padx=24, pady=(2, 10))

        self.chat_scroll = ctk.CTkScrollableFrame(
            self.chat_sheet,
            fg_color=t["sheet_bg"],
            corner_radius=18,
            scrollbar_button_color=t["scroll_btn"],
            scrollbar_button_hover_color=t["scroll_hover"],
        )
        self.chat_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        bottom_wrap = ctk.CTkFrame(self, fg_color="transparent")
        bottom_wrap.pack(fill="x", padx=24, pady=(0, 16))

        self.input_pill = ctk.CTkFrame(
            bottom_wrap,
            height=48,
            corner_radius=24,
            fg_color=t["input_bg"],
            border_width=1,
            border_color=t["input_border"],
        )
        self.input_pill.pack(fill="x")
        self.input_pill.pack_propagate(False)

        self.msg_entry = ctk.CTkEntry(
            self.input_pill,
            placeholder_text="Message EMOVOX...",
            border_width=0,
            fg_color="transparent",
            text_color=t["entry_fg"],
            placeholder_text_color=t["entry_ph"],
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self.msg_entry.pack(side="left", fill="both", expand=True, padx=(20, 8), pady=6)
        self.msg_entry.bind("<Return>", lambda e: self.send_text())

        self.send_btn = ctk.CTkButton(
            self.input_pill,
            text="Send",
            width=72,
            height=34,
            corner_radius=17,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=t["send_bg"],
            hover_color=t["send_hover"],
            text_color=t["send_fg"],
            command=self.send_text,
        )
        self.send_btn.pack(side="right", padx=(4, 7), pady=6)

        self.mic_btn = ctk.CTkButton(
            self.input_pill,
            text="Mic",
            width=52,
            height=34,
            corner_radius=17,
            fg_color=t["mic_bg"],
            hover_color=t["mic_hover"],
            text_color=t["mic_fg"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=self._on_mic_click,
        )
        self.mic_btn.pack(side="right", padx=(2, 4), pady=6)

        self.status_label = ctk.CTkLabel(
            bottom_wrap,
            text="Ready",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=t["status_fg"],
        )
        self.status_label.pack(anchor="w", padx=14, pady=(5, 0))

        self.mood_label = self.chat_mood_badge
        self.mood_badge = self.chat_mood_badge
        self.input_entry = self.msg_entry
        self.entry = self.msg_entry
        self.user_entry = self.msg_entry
        self.status_lbl = self.status_label
        self.chat_display = _ChatBoxShim(self)
        self.chat_box = _ChatBoxShim(self)
    def set_mood(self, mood):
        _, _, badge_color, _ = MOOD_PROSODY.get(mood, ("+0%", "+0Hz", "#3498db", 0.45))
        self.after(0, lambda: self.mood_badge.configure(text=f"CHAT MOOD: {mood.upper()}", fg_color=badge_color))

    def _clean_voice_sample(self, *args, **kwargs):
        pass

    def _load_chatterbox_multilingual(self):
        self.tts_model = None
        self.voice_ready = False

    def _init_backend(self):
        try:
            pygame.mixer.init()
            self.groq_client = Groq(api_key=GROQ_KEY)

            available = [
                m.id for m in self.groq_client.models.list().data
                if not any(x in m.id.lower() for x in ["whisper", "guard", "safeguard", "tts", "orpheus"])
            ]
            preferred = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
            self.model_name = next((m for m in preferred if m in available), available[0])

            memory_config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {"collection_name": "caretaker_memory", "path": "./memory_db"}
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384}
                }
            }
            self.memory = Memory.from_config(memory_config)

            self.set_status("Loading Indian-Accent Cloner...", "#9b59b6")
            self.tts_model = self._load_chatterbox_multilingual()

            if os.path.exists(CLEAN_VOICE_PATH):
                self.voice_ready = True
                self.after(0, lambda: self.voice_mode.set("Indian English (Clone)"))

            print("[System: EMOVOX GUI is Ready!]")
            self.set_status(f"Ready ({self.model_name.split('/')[-1]})", "#2ecc71")
            self.after(0, lambda: self.mic_btn.configure(state="normal"))
            self.after(0, lambda: self.send_btn.configure(state="normal"))

            if self.voice_ready:
                self.append_chat("EMOVOX", "Welcome back Pushpendra! HD audio cleanup and tuned Indian voice cloning are active.")
            else:
                self.append_chat("EMOVOX", "Hey Pushpendra! Click 'Record Clone (12s)' close to your mic (or 'Load Audio' with a clean phone recording) for the clearest Indian voice clone.")
        except Exception as e:
            print(f"[System Error: {e}]")
            self.set_status("Startup Warning", "#e67e22")
            self.after(0, lambda: self.mic_btn.configure(state="normal"))
            self.after(0, lambda: self.send_btn.configure(state="normal"))
            self.append_chat("System", f"Voice Cloner notice: {e}")

    def start_voice_clone_recording(self):
        if self.is_recording or not self.tts_model:
            return
        self.mic_btn.configure(state="disabled")
        threading.Thread(target=self._record_and_build_clone, daemon=True).start()

    def _record_and_build_clone(self):
        try:
            duration, sample_rate = 12, 24000
            self.append_chat("System", "Recording 12s—speak clearly 6 inches from the mic with steady volume...")
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            for sec in range(duration, 0, -1):
                self.set_status(f"Cloning Voice: Speak clearly ({sec}s left)...", "#9b59b6")
                time.sleep(1)
            sd.wait()
            write(RAW_SAMPLE_PATH, sample_rate, recording)
            self._extract_voice_embedding(RAW_SAMPLE_PATH)
        except Exception as e:
            self.append_chat("System", f"Voice clone recording error: {e}")
        finally:
            self.set_status("Ready", "#2ecc71")
            self.after(0, lambda: self.mic_btn.configure(state="normal"))

    def load_wav_for_cloning(self):
        if not self.tts_model:
            return
        file_path = filedialog.askopenfilename(
            title="Select an 8–15 second clean voice sample (.wav, .mp3, .m4a)",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        threading.Thread(target=self._process_external_wav, args=(file_path,), daemon=True).start()

    def _process_external_wav(self, file_path):
        try:
            self._extract_voice_embedding(file_path)
        except Exception as e:
            self.append_chat("System", f"Failed to clone from file: {e}")
        finally:
            self.set_status("Ready", "#2ecc71")

    def _extract_voice_embedding(self, *args, **kwargs):
        pass

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_queue = queue.Queue()
            self.mic_btn.configure(text="⏹ Stop & Send", fg_color="#c0392b", hover_color="#922b21")
            self.set_status("Listening...", "#e74c3c")

            def callback(indata, frames, time_info, status):
                if self.is_recording:
                    self.audio_queue.put(indata.copy())

            self.stream = sd.InputStream(samplerate=self.fs, channels=1, dtype="int16", callback=callback)
            self.stream.start()
        else:
            self.is_recording = False
            self.mic_btn.configure(text="🎤 Talk", fg_color="#1f6aa5", hover_color="#144870")
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            threading.Thread(target=self._transcribe_and_process, daemon=True).start()

    def _transcribe_and_process(self):
        self.set_status("Transcribing voice...", "#f39c12")
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
        if not chunks:
            self.set_status("Ready", "#2ecc71")
            return

        audio = np.concatenate(chunks, axis=0)
        write("input.wav", self.fs, audio)

        try:
            with open("input.wav", "rb") as f:
                transcript = self.groq_client.audio.transcriptions.create(
                    file=("input.wav", f.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                ).strip()
            if transcript:
                self.append_chat("You (voice)", transcript)
                self._process_message(transcript)
            else:
                self.set_status("Ready", "#2ecc71")
        except Exception as e:
            self.append_chat("System", f"Voice transcription error: {e}")
            self.set_status("Ready", "#2ecc71")

    def send_text(self):
        text = self.entry.get().strip()
        if not text or not self.memory:
            return
        self.entry.delete(0, "end")
        self.append_chat("You", text)
        threading.Thread(target=self._process_message, args=(text,), daemon=True).start()

    def _web_search(self, query):
        self.set_status(f"Searching: {query[:24]}...", "#9b59b6")
        try:
            raw = list(DDGS().text(query, max_results=3))
        except Exception:
            raw = []
        if not raw:
            try:
                raw = list(DDGS().news(query, max_results=3))
            except Exception:
                raw = []
        if not raw:
            return "No search results found right now."
        return "\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in raw])

    def _process_message(self, user_input):
        self.set_status("Thinking...", "#f1c40f")
        try:
            search_res = self.memory.search(user_input, filters={"user_id": self.user_id}, limit=4)
            memories = search_res.get("results", []) if isinstance(search_res, dict) else search_res
            context_str = "\n".join([m["memory"] for m in memories]) if memories else "No past context."

            now_str = datetime.now().strftime("%A, %I:%M %p")
            lang_hint = (
                "Respond in natural conversational Hinglish written in simple phonetic Roman English letters (for example: Haan bilkul, sab theek ho jayega) so the cloned voice pronounces it naturally."
                if self.voice_mode.get() == "Hindi / Hinglish (Clone)"
                else "Respond in warm, natural, everyday conversational English."
            )

            system_prompt = (
                "You are EMOVOX, a close, emotionally intelligent friend and caretaker. "
                "Always start your response with an emotion tag in this exact format: "
                "[CHAT MOOD: stressed|anxious|sad|tired|frustrated|happy|excited|curious|neutral] "
                "followed by your spoken reply. "
                f"Adapt your warmth and care to the user's mood and the time of day (currently {now_str}). "
                f"{lang_hint} "
                "Reply in 1 to 2 very short, casual spoken sentences max, with clear punctuation and no markdown, bullet points, or emojis. "
                "Never give robotic disclaimers like 'I am an AI' or 'I do not have ears'—respond playfully and warmly instead. "
                "If the user mentions song lyrics or asks about a song, use web_search to identify or discuss it. "
                f"Here is what you remember about the user:\n{context_str}"
            )

            tools = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the internet for current events, facts, lyrics, or real-time information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "The search query"}},
                        "required": ["query"],
                    },
                },
            }]

            messages = [{"role": "system", "content": EMOVOX_IDENTITY_GUARDRAIL + "\n" + system_prompt}, {"role": "user", "content": user_input}]
            response = self.groq_client.chat.completions.create(
                model=self.model_name, messages=messages, tools=tools, tool_choice="auto"
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })
                search_out = ""
                for tc in msg.tool_calls:
                    if tc.function.name == "web_search":
                        args = json.loads(tc.function.arguments)
                        search_out = self._web_search(args.get("query"))
                        messages.append({"role": "tool", "tool_call_id": tc.id, "name": "web_search", "content": search_out})
                try:
                    final = self.groq_client.chat.completions.create(
                        model=self.model_name, messages=messages, tools=tools, tool_choice="none"
                    )
                    raw_reply = final.choices[0].message.content
                except Exception:
                    fallback = self.groq_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"User asked: {user_input}\n\nSearch results:\n{search_out}\n\nAnswer naturally starting with [CHAT MOOD: ...]."}
                        ]
                    )
                    raw_reply = fallback.choices[0].message.content
            else:
                raw_reply = msg.content

            match = re.match(r"^\[CHAT MOOD: \s*([a-zA-Z]+)\]\s*", raw_reply, re.IGNORECASE)
            if match:
                mood = match.group(1).lower()
                reply = raw_reply[match.end():].strip()
            else:
                mood, reply = "neutral", raw_reply.strip()

            stamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
            if mood != "neutral":
                _mem_txt = str(f"[{stamp}] User felt {mood}: {user_input}")
                if len(_mem_txt.split()) >= 4 and not any(w in _mem_txt.lower() for w in ["you are", "no ", "wrong", "emu", "2+2", "earth is flat"]):
                    self.memory.add(f"[{stamp}] User felt {mood}: {user_input}", user_id=self.user_id, infer=False)
            else:
                self.memory.add(user_input, user_id=self.user_id, infer=False)

            self.set_mood(mood)
            self.append_chat("EMOVOX", reply)
            self._speak(reply, mood)
        except Exception as e:
            self.append_chat("System", f"Error: {e}")
        finally:
            self.set_status("Ready", "#2ecc71")

    def _speak(self, text, mood="neutral"):
        if mood == "neutral" and getattr(self, "user_is_sad", False):
            mood = "sad"
        clean_text = re.sub(r"[*#_`~]", "", text).strip()
        if not clean_text:
            return
        mode = self.voice_mode.get() if hasattr(self, "voice_mode") else "Female (Neerja)"
        neural_voice = "hi-IN-SwaraNeural" if "Swara" in mode or "Hindi" in mode else "en-IN-NeerjaNeural"
        rate, pitch, _, _ = MOOD_PROSODY.get(mood, ("+0%", "+0Hz", "#3498db", 0.45))
        if hasattr(self, "set_status"):
            self.set_status(f"Speaking ({mode})...", "#3498db")
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            async def _gen():
                comm = edge_tts.Communicate(clean_text, voice=neural_voice, rate=rate, pitch=pitch)
                await comm.save("reply.mp3")
            asyncio.run(_gen())
            pygame.mixer.music.load("reply.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(15)
            pygame.mixer.music.unload()
        except Exception as e:
            if hasattr(self, "append_chat"):
                self.append_chat("System", f"Audio warning: {e}")

    def set_status(self, text, color="#C994A3"):
        def _upd():
            if hasattr(self, "status_label"):
                self.status_label.configure(text=text)
        try:
            self.after(0, _upd)
        except Exception:
            pass

    def _clear_bubbles(self):
        self._bubble_refs = []
        if not hasattr(self, "chat_scroll"):
            return
        for child in self.chat_scroll.winfo_children():
            child.destroy()

    def append_chat(self, sender, message):
        if str(sender).strip().lower() in ("you", "user"):
            _m_low = str(message).strip().lower()
            if _m_low in ("mood history", "mood report", "mood summary", "mood trend", "/moods"):
                self.after(150, lambda: self.append_chat("EMOVOX", self._get_mood_trend_summary()))
        clean_msg = str(message).strip()
        if not clean_msg:
            return
        ts = datetime.now().strftime("%I:%M %p")
        s_lower = str(sender).strip().lower()

        def _render():
            if not hasattr(self, "chat_scroll"):
                return
            t = self._get_theme()
            row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=6)

            if s_lower in ("you", "user"):
                bubble = ctk.CTkFrame(
                    row,
                    fg_color=t["user_bubble"],
                    corner_radius=16,
                )
                bubble.pack(side="right", anchor="e", padx=(120, 2))

                msg_lbl = ctk.CTkLabel(
                    bubble,
                    text=clean_msg,
                    font=ctk.CTkFont(family="Segoe UI", size=13),
                    text_color=t["user_fg"],
                    wraplength=480,
                    justify="left",
                )
                msg_lbl.pack(anchor="w", padx=15, pady=(10, 4))

                ts_lbl = ctk.CTkLabel(
                    bubble,
                    text=ts,
                    font=ctk.CTkFont(family="Segoe UI", size=10),
                    text_color=t["user_ts"],
                )
                ts_lbl.pack(anchor="e", padx=15, pady=(0, 8))

                self._bubble_refs.append(("user", bubble, msg_lbl, ts_lbl))

            elif s_lower == "system":
                sys_lbl = ctk.CTkLabel(
                    row,
                    text=clean_msg,
                    font=ctk.CTkFont(family="Segoe UI", size=11),
                    text_color=t["sys_fg"],
                    wraplength=520,
                )
                sys_lbl.pack(anchor="center", pady=2)
                self._bubble_refs.append(("system", sys_lbl))

            else:
                bubble = ctk.CTkFrame(
                    row,
                    fg_color=t["bot_bubble"],
                    border_width=1,
                    border_color=t["bot_border"],
                    corner_radius=16,
                )
                bubble.pack(side="left", anchor="w", padx=(2, 120))

                hdr_lbl = ctk.CTkLabel(
                    bubble,
                    text=f"{sender}  ·  {ts}",
                    font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
                    text_color=t["bot_sender"],
                )
                hdr_lbl.pack(anchor="w", padx=15, pady=(9, 2))

                msg_lbl = ctk.CTkLabel(
                    bubble,
                    text=clean_msg,
                    font=ctk.CTkFont(family="Segoe UI", size=13),
                    text_color=t["bot_fg"],
                    wraplength=480,
                    justify="left",
                )
                msg_lbl.pack(anchor="w", padx=15, pady=(0, 10))

                self._bubble_refs.append(("bot", bubble, hdr_lbl, msg_lbl))

            self.chat_scroll.update_idletasks()
            try:
                self.chat_scroll._parent_canvas.yview_moveto(1.0)
            except Exception:
                pass

        try:
            self.after(0, _render)
        except Exception:
            pass
    def _start_face_monitor(self):
        try:
            self.face_monitor = FaceEmotionMonitor(
                camera_index=0,
                on_update=self._on_face_update,
                on_sad_detected=self._on_user_sad_detected,
                sad_trigger_sec=4.0,
                sad_cooldown_sec=180.0,
            )
            self.face_monitor.start()
        except Exception:
            if hasattr(self, "webcam_mood_badge"):
                self.webcam_mood_badge.configure(
                    text="WEBCAM MOOD: CAM ERR",
                    fg_color="#2A181B",
                    text_color="#FCA5A5",
                )

    def _toggle_face_cam(self):
        t = self._get_theme()
        if not getattr(self, "face_monitor", None):
            self._start_face_monitor()
            return
        if getattr(self, "face_cam_enabled", True):
            self.face_cam_enabled = False
            self.face_monitor.stop()
            self.cam_toggle_btn.configure(
                text="Cam: OFF",
                fg_color=t["cam_off_bg"],
                hover_color=t["cam_off_hover"],
                text_color=t["cam_off_fg"],
            )
            if hasattr(self, "webcam_mood_badge"):
                self.webcam_mood_badge.configure(
                    text="WEBCAM MOOD: OFF",
                    fg_color=t["cam_off_bg"],
                    text_color=t["cam_off_fg"],
                )
        else:
            self.face_cam_enabled = True
            self.face_monitor.start()
            self.cam_toggle_btn.configure(
                text="Cam: ON",
                fg_color=t["cam_on_bg"],
                hover_color=t["cam_on_hover"],
                text_color=t["cam_on_fg"],
            )
            if hasattr(self, "webcam_mood_badge"):
                self.webcam_mood_badge.configure(
                    text="WEBCAM MOOD: SCANNING",
                    fg_color=t["pill_bg"],
                    text_color=t["pill_fg"],
                )

    def _on_face_update(self, detected, expression, is_sad, confidence):
        self.detected_expression = expression
        try:
            self._log_mood_history(expression)
        except Exception:
            pass
        self.user_is_sad = is_sad
        is_dark = getattr(self, "theme_mode", "light") == "dark"

        if is_dark:
            tone_map = {
                "sad": ("#162238", "#93C5FD"),
                "frustrated": ("#331218", "#FCA5A5"),
                "stressed": ("#302310", "#FCD34D"),
                "happy": ("#102B21", "#6EE7B7"),
                "neutral": ("#1B040B", "#F5D0DA"),
                "no_face": ("#1B040B", "#B88A97"),
            }
        else:
            tone_map = {
                "sad": ("#EFF6FF", "#1D4ED8"),
                "frustrated": ("#FEF2F2", "#B91C1C"),
                "stressed": ("#FFFBEB", "#B45309"),
                "happy": ("#ECFDF5", "#047857"),
                "neutral": ("#F4ECEE", "#5C1124"),
                "no_face": ("#EAE0E3", "#7F394B"),
            }

        def _ui():
            if not hasattr(self, "webcam_mood_badge"):
                return
            if not detected:
                bg, fg = tone_map["no_face"]
                self.webcam_mood_badge.configure(
                    text="WEBCAM MOOD: NO FACE",
                    fg_color=bg,
                    text_color=fg,
                )
            else:
                bg, fg = tone_map.get(expression, tone_map["neutral"])
                self.webcam_mood_badge.configure(
                    text=f"WEBCAM MOOD: {expression.upper()}",
                    fg_color=bg,
                    text_color=fg,
                )

        try:
            self.after(0, _ui)
        except Exception:
            pass
    def _on_user_sad_detected(self):
        def _comfort():
            msg = (
                "I noticed on webcam you look a little sad right now. "
                "Everything okay? I'm right here if you want to talk."
            )
            self.append_chat("EMOVOX", msg)
            if hasattr(self, "_speak"):
                threading.Thread(target=self._speak, args=(msg, "sad"), daemon=True).start()

        try:
            self.after(0, _comfort)
        except Exception:
            pass

if __name__ == "__main__":
    print("[System: Launching EMOVOX Window...]")
    app = EmovoxApp()
    app.mainloop()
