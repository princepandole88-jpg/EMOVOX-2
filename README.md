# EMOVOX — Real-Time Multimodal Emotion-Aware AI Companion

**EMOVOX** is a desktop AI companion that perceives facial expressions in real time, analyzes conversational sentiment, adapts its vocal prosody and responses to the user's mood, and retains persistent long-term memory across sessions.

**Created by:** Pushpendra Pandole & Puneet Kushwaha

---

## Key Features

- **Dual-Engine Vision Pipeline (`face_emotion.py`):**
  - Uses **MediaPipe 468-point Face Mesh** geometric ratios (Eye Aspect Ratio, Mouth Aspect Ratio, brow/lip landmarks) for lightweight, CPU-friendly emotion recognition without heavy deep-learning overhead.
  - Includes an **OpenCV Haar-Cascade + Sobel** low-light fallback with **CLAHE** contrast equalization and a **12-frame rolling majority-vote smoothing window** to prevent mood flickering.
- **Multimodal Emotion Fusion:** Combines real-time `WEBCAM MOOD` and `CHAT MOOD` to tailor empathy and tone dynamically.
- **Voice Input & Expressive TTS (`emovox_gui.py`):**
  - Speech-to-Text via `SpeechRecognition` with phonetic regex normalization (`_clean_stt_text`).
  - Natural neural Text-to-Speech via `edge-tts` (`Neerja` and `Swara` voices) with mood-matched rate and pitch prosody.
- **Persistent Long-Term Memory:** Powered by **Mem0** + local **ChromaDB** vector storage so EMOVOX remembers user preferences, facts, and conversation context across restarts.
- **Modern Desktop UI:** Built with **CustomTkinter** featuring live mood badges, dark/maroon themes, voice switching, and local mood history tracking.

---

## Project Structure

```text
EMOVOX-2/
├── emovox_gui.py       # Main CustomTkinter GUI, Groq LLM integration, STT/TTS & Mem0 memory
├── face_emotion.py     # Real-time MediaPipe Face Mesh + OpenCV CLAHE emotion detector
└── requirements.txt    # Python dependencies
