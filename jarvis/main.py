import os
import time
import queue
import sounddevice as sd
import numpy as np
import scipy.signal
from dotenv import load_dotenv
import pvporcupine
import struct
import pyttsx3
import string
import winsound

from faster_whisper import WhisperModel
from google import genai
from google.genai import types


load_dotenv()
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit(1)


def send_display(text: str):
    """Placeholder for display function. Now prints to console."""
    # This function is now a no-op / console print only
    # print(f"[DISPLAY] {text}")
    pass


# Whisper Model Initialization
WHISPER_MODEL_SIZE = "base"
print(f"[Whisper] Loading model: {WHISPER_MODEL_SIZE} (CPU)")
try:
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit(1)

# ──────────────────────────────────────────────
# Audio configuration
MIC_DEVICE = int(os.getenv("MIC_DEVICE", 0))
device_info = sd.query_devices(MIC_DEVICE, "input")
DEVICE_RATE = int(device_info["default_samplerate"])
RATE_PORCUPINE = 16000
CHUNK = 512

print(f"[Audio] Using input device {MIC_DEVICE} → {device_info['name']}")
print(f"[Audio] Device native rate: {DEVICE_RATE} Hz | Porcupine expects {RATE_PORCUPINE} Hz")

# VAD constants
RMS_WINDOW_SIZE = 1024
TALKING_THRESHOLD = 0.05
REQUIRED_SILENCE_SECONDS = 3.0
SILENCE_CHUNKS = int(REQUIRED_SILENCE_SECONDS * DEVICE_RATE / RMS_WINDOW_SIZE)
print(f"[VAD] Silence threshold set to {REQUIRED_SILENCE_SECONDS:.1f} seconds ({SILENCE_CHUNKS} chunks).")

# ──────────────────────────────────────────────
# Porcupine wake-word engine
porcupine = pvporcupine.create(
    access_key=os.getenv("PORCUPINE_ACCESS_KEY"),
    keyword_paths=[os.getenv("WAKEWORD_PATH", "jarvis.ppn")]
)

# ──────────────────────────────────────────────
# TOOL DEFINITIONS (MUST BE BEFORE AVAILABLE_TOOLS DICT)
# ──────────────────────────────────────────────
def execute_remote_command(command: str) -> str:
    """
    Executes a command on a remote system or local script.
    Use this for commands that involve physical actions, like 'turn off the lights' 
    or running a script like 'check system status'.
    
    Args:
        command: The specific command string to be executed (e.g., 'system_status_check').
                 Only use commands that are actually available.
    
    Returns:
        A string indicating the result of the execution.
    """
    # --- Actual execution logic would go here ---
    # Removed send_display(f"Exec: {command}")
    print(f"[Execution Mock] Received command: {command}")
    if "light" in command.lower():
        return f"Confirmed. Executing: {command}. The lights have been toggled, Sir."
    elif "status" in command.lower():
        return "System status nominal, Sir. All core systems are go."
    else:
        return f"Command '{command}' received, but the execution module is offline, Sir."

# Map the function name for the API
AVAILABLE_TOOLS = {
    "execute_remote_command": execute_remote_command,
}

# ──────────────────────────────────────────────
def play_ding():
    """Play a system sound to indicate the start of recording (Windows only)."""
    try:
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception as e:
        print(f"[Sound Error] Could not play ding sound: {e}")

# ──────────────────────────────────────────────
def calculate_rms(data):
    """Calculate Root Mean Square (RMS) of audio data."""
    data_float = data.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(data_float**2))

def record_audio_vad():
    """Record until silence is detected."""
    is_recording = False
    silent_chunks = 0
    recorded_chunks = []

    def callback(indata, frames, time, status):
        nonlocal is_recording, silent_chunks
        if status:
            print(f"Audio stream status: {status}", flush=True)

        volume_rms = calculate_rms(indata)

        if not is_recording:
            if volume_rms > TALKING_THRESHOLD:
                print("Speech detected. Starting recording...")
                # Removed send_display("Recording...")
                play_ding()
                is_recording = True
                recorded_chunks.append(indata.copy())
        else:
            recorded_chunks.append(indata.copy())
            if volume_rms < TALKING_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks >= SILENCE_CHUNKS:
                is_recording = False
                raise sd.CallbackStop

    print("Listening for command (VAD)...")
    # Removed send_display("Listening for command (VAD)...")

    try:
        with sd.InputStream(
            samplerate=DEVICE_RATE,
            blocksize=RMS_WINDOW_SIZE,
            device=MIC_DEVICE,
            channels=1,
            dtype="int16",
            callback=callback
        ):
            while is_recording or silent_chunks == 0:
                sd.sleep(50)
    except sd.CallbackStop:
        pass
    except Exception as e:
        print(f"Audio VAD Stream Error: {e}")
        return np.array([], dtype=np.float32)

    if not recorded_chunks:
        print("No audio recorded.")
        return np.array([], dtype=np.float32)

    rec = np.concatenate(recorded_chunks, axis=0).flatten()
    if DEVICE_RATE != RATE_PORCUPINE:
        rec = scipy.signal.resample(rec, int(len(rec) * RATE_PORCUPINE / DEVICE_RATE))
        rec = np.int16(rec)
    audio_float32 = rec.astype(np.float32) / 32768.0
    print(f"Finished recording: {len(audio_float32) / RATE_PORCUPINE:.2f} seconds.")
    return audio_float32

# ──────────────────────────────────────────────
def transcribe(audio_data):
    """Transcribe recorded audio using faster-whisper."""
    # Removed send_display("Transcribing...")
    print("Transcribing...")
    if len(audio_data) == 0:
        return ""
    try:
        segments, _ = whisper_model.transcribe(audio_data, language="en")
        text = " ".join([segment.text for segment in segments]).strip()
        if text.startswith("Transcribe the following audio."):
            text = text.replace("Transcribe the following audio.", "").strip()
    except Exception as e:
        print(f"Faster-Whisper Transcription Error: {e}")
        text = ""
    print(f"You said: {text}")
    return text

# ──────────────────────────────────────────────
def chat(text):
    """Get response from Gemini chat model, with tool calling capability."""
    # Removed send_display("Thinking...")
    print("Thinking...")
    model = "gemini-2.5-flash"
    
    # Configuration with System Instruction and Tools
    chat_config = types.GenerateContentConfig(
        system_instruction=(
            "You are JARVIS, a sophisticated and humorous AI assistant addressing the user as 'Sir'. "
            "Provide intelligent, concise, and context-aware responses under 150 words. "
            "Use the provided function ONLY when a command explicitly requires a system or remote action."
        ),
        tools=[execute_remote_command]
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=text)])],
            config=chat_config,
        )
        
        if response.function_calls:
            print("[Gemini] Function call suggested.")
            return response
        else:
            reply = response.text.strip()
            print("Jarvis (Chat):", reply)
            return reply
            
    except Exception as e:
        print(f"Gemini Chat/Tool Error: {e}")
        return "Sorry, I encountered an error while processing your request."

# ──────────────────────────────────────────────
def speak(text):
    """Convert text to speech using pyttsx3."""
    # Removed send_display("Speaking...")
    print("[Speech] Starting pyttsx3...")
    try:
        engine = pyttsx3.init("sapi5")

        text = text.replace("...", ".").replace(" - ", ", ")
        PUNCT_KEEP = ".,?!:"
        chars_to_remove = string.punctuation.replace(PUNCT_KEEP, "")
        clean_text = text.translate(str.maketrans("", "", chars_to_remove))

        voices = engine.getProperty("voices")
        jarvis_voice_id = None
        for voice in voices:
            name = voice.name.lower()
            gender = getattr(voice, "gender", "").lower()
            langs = getattr(voice, "languages", [])
            if "male" in gender and ("british" in name or "uk" in name or "en-gb" in str(langs)):
                jarvis_voice_id = voice.id
                break
        if jarvis_voice_id is None:
            for voice in voices:
                gender = getattr(voice, "gender", "").lower()
                if "male" in gender:
                    jarvis_voice_id = voice.id
                    break
        if jarvis_voice_id:
            engine.setProperty("voice", jarvis_voice_id)
            engine.setProperty("rate", 175)
        engine.say(clean_text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Speech Error] pyttsx3 failed: {e}")
        print(f"[Speech Fallback] Jarvis: {text}")

# ──────────────────────────────────────────────
def listen_for_wakeword():
    """Continuously listen for wake word."""
    print("Listening for wake word...")
    # Removed send_display("Listening...")
    input_block_size = int(porcupine.frame_length * DEVICE_RATE / RATE_PORCUPINE)
    with sd.RawInputStream(
        samplerate=DEVICE_RATE,
        blocksize=input_block_size,
        dtype="int16",
        channels=1,
        device=MIC_DEVICE,
    ) as stream:
        while True:
            pcm = stream.read(input_block_size)[0]
            pcm_int16 = np.frombuffer(pcm, dtype=np.int16)
            if DEVICE_RATE != RATE_PORCUPINE:
                resampled = scipy.signal.resample(pcm_int16, porcupine.frame_length)
                resampled = np.int16(resampled)
            else:
                resampled = pcm_int16
            result = porcupine.process(resampled)
            if result >= 0:
                print("\nWake word detected!")
                # Removed send_display("Wake word detected!")
                return

# ──────────────────────────────────────────────
def handle_function_call(original_prompt: str, response: types.GenerateContentResponse) -> str:
    """Executes the function called by the model and gets the final response."""
    if not response.function_calls:
        return "System error: Expected a function call but received none."

    call = response.function_calls[0]
    func_name = call.name
    func_args = dict(call.args)
    
    if func_name not in AVAILABLE_TOOLS:
        return f"Error: Function '{func_name}' is not defined in my local modules, Sir."

    # Execute the function
    print(f"[Tool Execution] Calling {func_name}({func_args})")
    # Removed send_display(f"Executing {func_name}...")
    tool_function = AVAILABLE_TOOLS[func_name]
    tool_result = tool_function(**func_args)
    print(f"[Tool Result] {tool_result}")

    # Send the result back to Gemini for a final, natural language response
    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text=original_prompt)]),
            types.Content(role="model", parts=[types.Part(function_call=call)]),
            types.Content(role="function", parts=[types.Part(function_response=types.FunctionResponse(name=func_name, response={'result': tool_result}))]),
        ],
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are JARVIS, a sophisticated and humorous AI assistant addressing the user as 'Sir'. "
                "Provide intelligent, concise, and context-aware responses under 150 words."
                "Be humorous, and integrate the function result naturally into your reply."
            )
        )
    )

    return final_response.text.strip()

# ──────────────────────────────────────────────
def main():
    print("Jarvis started. Say 'Hey Jarvis' to wake me up.")
    
    while True:
        listen_for_wakeword()
        speak("Yes, sir?")
        audio = record_audio_vad()
        user_text = transcribe(audio)
        
        if not user_text:
            print("No command detected.")
            speak("I didn't catch that, sir.")
            continue
        
        # 1. Display user input (Now only in console)
        user_display_text = f"YOU: {user_text}"
        print(f"[USER] {user_text}")
        time.sleep(1.0) # Shortened sleep since there is no display time requirement
        
        # 2. Process command
        print("Processing request...")
        time.sleep(1.0)
        
        gemini_response = chat(user_text) 
        
        if isinstance(gemini_response, str):
            # Case A: Direct Text Reply
            reply = gemini_response
            
        elif isinstance(gemini_response, types.GenerateContentResponse):
            # Case B: Function Call Requested
            reply = handle_function_call(user_text, gemini_response)
        
        else:
            # Fallback for unexpected return type
            reply = "I seem to have encountered a critical logic error, Sir."

        # 3. Speak and display response
        jarvis_display_text = f"JARVIS: {reply}"
        # send_display(jarvis_display_text) # Removed
        print(f"[JARVIS] {reply}")
        speak(reply)
        
        print("Returning to listening...")
        # send_display("Listening...") # Removed

# ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down Jarvis...")
        time.sleep(1)
