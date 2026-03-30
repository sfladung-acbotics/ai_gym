import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import webrtcvad
import collections
import threading

import contextlib
import sys
import os
import queue
import ollama

from piper import PiperVoice


class Voice_Parse_Agent:
    def __init__(self, tts=None):
        self.commands = queue.Queue()
        self.tts = tts

    def add_speech(self, speech, subject):
        self.commands.put((speech, subject))

    def run_once(self):
        try:
            command, subject = self.commands.get(timeout=1)
        except queue.Empty:
            return
        # 2. Start the chat with the tool enabled

        # todo: should we include context/field info in the command to hint the preprocessor?
        messages = [
            {
                "role": "system",
                "content": "You are a preprocessor for a text to speech system. You will simplify the following. You will remove extraneous formatting such as *. You will expand abreviations into full words. You will rephrase the output to be concise. For determining proper abreviation expansions, use the following hints about the subject. %s"
                % (subject,),
            },
            {"role": "user", "content": command},
        ]
        print("...")
        response = ollama.chat(
            model="qwen3:1.7b",
            messages=messages,
            think=False,
            # options={
            #     "temperature": 0,  # 0 is the most deterministic
            #     "seed": 42,  # Optional: lock the random seed for even more consistency
            # },
        )
        print(response)
        print(response.message.content)
        if self.tts:
            self.tts.speak_async(response.message.content)

    def run(self):
        while True:
            self.run_once()

    def run_as_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()


class TTS_Engine:
    def __init__(self, model_path, config_path=None):
        """
        model_path: path to the .onnx file
        config_path: path to the .onnx.json (defaults to model_path + .json)
        """
        self.voice = PiperVoice.load(model_path, config_path=config_path)
        with ignore_stderr():
            self.pa = pyaudio.PyAudio()
        self.rate = self.voice.config.sample_rate  # Usually 22050Hz

    def speak(self, text):
        if not text:
            return

        print(f"[TTS] {text}")

        # Open a stream for the specific sample rate of the voice

        stream = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=self.rate, output=True
        )

        for chunk in self.voice.synthesize(text):
            # 1. chunk.audio is a numpy array (int16)
            # 2. .tobytes() converts it to raw PCM for pyaudio
            audio_bytes = (
                chunk.audio_int16_bytes
            )  # np.array(chunk.audio_int16_bytes, dtype=np.int16).tobytes()
            stream.write(audio_bytes)

        stream.stop_stream()
        stream.close()

    def speak_async(self, text):
        """Runs the speak method in a separate thread so the agent doesn't hang."""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()


def pcm_to_float(data):
    """Utility to convert raw bytes to normalized float32."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


@contextlib.contextmanager
def ignore_stderr():
    """Redirects stderr to devnull to hide ALSA/JACK warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


class Voice_Agent:
    def __init__(self, output_callbacks=()):
        # VAD
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30  # VAD requires 10, 20, or 30ms
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.VAD_SENSITIVITY = 3  # 0 (least aggressive) to 3 (most aggressive)

        # Initialize VAD and Whisper
        self.vad = webrtcvad.Vad(self.VAD_SENSITIVITY)
        self.output_callbacks = list(output_callbacks)

        # --- Initialization ---
        self.model_size = "tiny.en"
        self.model = WhisperModel(
            self.model_size, device="cuda", compute_type="float16"
        )
        self.audio_queue = queue.Queue()

    def run_mic_injest(self):
        with ignore_stderr():
            p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=480,
        )  # 30ms frames

        print("[SYSTEM] Microphone is now hot.")
        while True:
            try:
                chunk = stream.read(480, exception_on_overflow=False)
                self.audio_queue.put(chunk)
            except Exception as e:
                print(f"Mic error: {e}")

    def run_audio_process(self):
        vad = webrtcvad.Vad(2)
        voiced_frames = []
        triggered = False
        silence_counter = 0  # Use a ring buffer to keep a small amount of audio *before* speech is detected
        pre_roll = collections.deque(maxlen=10)  # Keep ~300ms of history
        while True:
            # Get a 30ms chunk from the background thread
            frame = self.audio_queue.get()
            is_speech = vad.is_speech(frame, 16000)

            if not triggered:
                pre_roll.append(frame)  # Keep filling while quiet
                if is_speech:
                    triggered = True
                    voiced_frames.extend(list(pre_roll))
                    pre_roll.clear()
            else:
                voiced_frames.append(frame)
                if not is_speech:
                    silence_counter += 1
                else:
                    silence_counter = 0

                # If speech ends, process it!
                if silence_counter > 33:  # ~1 second of silence
                    audio_np = pcm_to_float(b"".join(voiced_frames))
                    segments, _ = self.model.transcribe(audio_np, beam_size=5)
                    user_text = " ".join([s.text for s in segments]).strip()

                    if user_text:
                        print(f"You: {user_text}")
                        for cb in self.output_callbacks:
                            cb(user_text)

                    triggered = False
                    voiced_frames = []
                    silence_counter = 0

    def run(self):
        self.input_thread = threading.Thread(target=self.run_mic_injest)
        self.process_thread = threading.Thread(target=self.run_audio_process)
        self.input_thread.start()
        self.process_thread.start()

    def wait_join(self):
        self.input_thread.join()
        self.process_thread.join()


if __name__ == "__main__":
    va = Voice_Agent()
    va.run()
