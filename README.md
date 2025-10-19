# Voice Assistant with Gemini, Faster Whisper, and Porcupine (Console Version)

This project implements a sophisticated voice assistant similar to JARVIS., running entirely on your local machine using Python, powered by **Google Gemini** for intelligence and **Faster Whisper** for transcription.

This version is streamlined to use **console output and text-to-speech (TTS)**, as the optional Arduino serial display functionality has been removed.

## Features

* **Wake Word Detection:** Uses **Porcupine** to listen for a custom wake word (e.g., "Hey Jarvis") entirely offline.
* **Voice Activity Detection (VAD):** Automatically detects when you stop speaking to end the recording.
* **Speech-to-Text (STT):** Uses **Faster Whisper** to quickly and accurately transcribe your voice commands.
* **Language Model (LLM):** Uses **Gemini 2.5 Flash** for intelligent chat responses.
* **Tool Calling:** Integrates a mock remote command execution function, demonstrating Gemini's ability to trigger system actions.
* **Text-to-Speech (TTS):** Uses **pyttsx3** to speak the AI's response in a "Jarvis-like" voice.

## Prerequisites

### 1. Python and Dependencies

You must have **Python 3.9+** installed. This project relies on the following Python libraries:

```bash
pip install -r requirements.txt
