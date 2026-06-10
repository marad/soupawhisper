#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper.
Hold the hotkey to record, release to transcribe and copy to clipboard.
"""

import argparse
import configparser
import json
import platform
import subprocess
import tempfile
import threading
import signal
import sys
import os
from pathlib import Path

from pynput import keyboard
from faster_whisper import WhisperModel

__version__ = "0.1.0"

IS_MACOS = platform.system() == "Darwin"

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"


def load_config():
    config = configparser.ConfigParser()

    # Defaults
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "language": "",
        "key": "f12",
        "auto_type": "true",
        "notifications": "true",
    }

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        "language": config.get("whisper", "language", fallback=defaults["language"]),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
    }


CONFIG = load_config()


def get_default_input_device():
    """Name of the system default audio input device, or None if unknown."""
    if not IS_MACOS:
        return None  # arecord uses the ALSA "default" device
    try:
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPAudioDataType", "-json"],
            capture_output=True, text=True, timeout=15,
        )
        items = json.loads(result.stdout)["SPAudioDataType"][0]["_items"]
        for item in items:
            if item.get("coreaudio_default_audio_input_device") == "spaudio_yes":
                return item["_name"]
    except Exception as e:
        print(f"Could not determine input device: {e}")
    return None


def print_input_device():
    """Print the current default input device (slow query, run in a thread)."""
    device = get_default_input_device()
    if device:
        print(f"Recording from: {device}")
    else:
        print("Recording from: system default input (could not resolve name)")


def get_hotkey(key_name):
    """Map key name to pynput key."""
    key_name = key_name.lower()
    if hasattr(keyboard.Key, key_name):
        return getattr(keyboard.Key, key_name)
    elif len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)
    else:
        print(f"Unknown key: {key_name}, defaulting to f12")
        return keyboard.Key.f12


HOTKEY = get_hotkey(CONFIG["key"])
# macOS virtual keycode of the hotkey; None for character keys (unknown keycode)
HOTKEY_VK = HOTKEY.value.vk if isinstance(HOTKEY, keyboard.Key) else HOTKEY.vk
MODEL_SIZE = CONFIG["model"]
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
LANGUAGE = CONFIG["language"] or None  # None = auto-detect
AUTO_TYPE = CONFIG["auto_type"]
NOTIFICATIONS = CONFIG["notifications"]


class Dictation:
    def __init__(self):
        # Serializes start/stop now that they run on their own threads
        self.lock = threading.Lock()
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True

        # Load model in background
        print(f"Loading Whisper model ({MODEL_SIZE})...")
        threading.Thread(target=self._load_model, daemon=True).start()

        # system_profiler takes a couple of seconds, don't block startup
        threading.Thread(target=print_input_device, daemon=True).start()

    def _load_model(self):
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            self.model_loaded.set()
            hotkey_name = HOTKEY.name if hasattr(HOTKEY, 'name') else HOTKEY.char
            print(f"Model loaded. Ready for dictation!")
            print(f"Hold [{hotkey_name}] to record, release to transcribe.")
            print("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            print(f"Failed to load model: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                print("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification."""
        if not NOTIFICATIONS:
            return
        if IS_MACOS:
            # Pass title/message via argv to avoid AppleScript string escaping
            subprocess.run(
                [
                    "osascript",
                    "-e", "on run argv",
                    "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
                    "-e", "end run",
                    title,
                    message
                ],
                capture_output=True
            )
            return
        subprocess.run(
            [
                "notify-send",
                "-a", "SoupaWhisper",
                "-i", icon,
                "-t", str(timeout),
                "-h", "string:x-canonical-private-synchronous:soupawhisper",
                title,
                message
            ],
            capture_output=True
        )

    def start_recording(self):
        if self.recording or self.model_error:
            return

        self.recording = True
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        if IS_MACOS:
            # Record using sox's rec (CoreAudio default input)
            record_cmd = [
                "rec",
                "-q",
                "-b", "16",      # 16-bit
                "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
                "-c", "1",       # Mono
                self.temp_file.name
            ]
        else:
            # Record using arecord (ALSA) - works on most Linux systems
            record_cmd = [
                "arecord",
                "-f", "S16_LE",  # Format: 16-bit little-endian
                "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
                "-c", "1",       # Mono
                "-t", "wav",
                self.temp_file.name
            ]
        self.record_process = subprocess.Popen(
            record_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("Recording...")
        hotkey_name = HOTKEY.name if hasattr(HOTKEY, 'name') else HOTKEY.char
        self.notify("Recording...", f"Release {hotkey_name.upper()} when done", "audio-input-microphone", 30000)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            if IS_MACOS:
                # sox finalizes the WAV header only on SIGINT, not SIGTERM
                self.record_process.send_signal(signal.SIGINT)
            else:
                self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        print("Transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 30000)

        # Wait for model if not loaded yet
        self.model_loaded.wait()

        if self.model_error:
            print(f"Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        # Transcribe
        try:
            segments, info = self.model.transcribe(
                self.temp_file.name,
                beam_size=5,
                vad_filter=True,
                language=LANGUAGE,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            if text:
                # Copy to clipboard
                clipboard_cmd = ["pbcopy"] if IS_MACOS else ["xclip", "-selection", "clipboard"]
                process = subprocess.Popen(clipboard_cmd, stdin=subprocess.PIPE)
                process.communicate(input=text.encode())

                # Type it into the active input field
                if AUTO_TYPE:
                    if IS_MACOS:
                        # Pass text via argv to avoid AppleScript string escaping;
                        # requires Accessibility permission for the terminal app
                        subprocess.run([
                            "osascript",
                            "-e", "on run argv",
                            "-e", 'tell application "System Events" to keystroke (item 1 of argv)',
                            "-e", "end run",
                            text
                        ])
                    else:
                        subprocess.run(["xdotool", "type", "--clearmodifiers", text])

                print(f"Copied: {text}")
                self.notify("Copied!", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                print("No speech detected")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            print(f"Error: {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            # Cleanup temp file
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)

    def _run_locked(self, fn):
        with self.lock:
            fn()

    def _spawn(self, fn):
        # The listener callback must return quickly: with the intercepting
        # event tap on macOS a blocked callback stalls keyboard input
        # system-wide until macOS disables the tap. The lock keeps
        # start/stop serialized as they were when run on the callback thread.
        threading.Thread(target=self._run_locked, args=(fn,), daemon=True).start()

    def on_press(self, key):
        if key == HOTKEY:
            self._spawn(self.start_recording)

    def on_release(self, key):
        if key == HOTKEY:
            self._spawn(self.stop_recording)

    def stop(self):
        print("\nExiting...")
        self.running = False
        os._exit(0)

    def run(self):
        listener_kwargs = {
            "on_press": self.on_press,
            "on_release": self.on_release,
        }
        if IS_MACOS and HOTKEY_VK is not None:
            import Quartz

            def suppress_hotkey(event_type, event):
                # Swallow the push-to-talk key so other applications never
                # see it; pynput dispatches on_press/on_release before this
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                return None if keycode == HOTKEY_VK else event

            listener_kwargs["darwin_intercept"] = suppress_hotkey

        with keyboard.Listener(**listener_kwargs) as listener:
            listener.join()


def check_dependencies():
    """Check that required system commands are available."""
    missing = []

    if IS_MACOS:
        # pbcopy and osascript are built into macOS; only sox is external
        required = [("rec", "sox")]
        install_hint = "brew install"
    else:
        required = [("arecord", "alsa-utils"), ("xclip", "xclip")]
        if AUTO_TYPE:
            required.append(("xdotool", "xdotool"))
        install_hint = "sudo apt install"

    for cmd, pkg in required:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            missing.append((cmd, pkg))

    if missing:
        print("Missing dependencies:")
        for cmd, pkg in missing:
            print(f"  {cmd} - install with: {install_hint} {pkg}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SoupaWhisper - Push-to-talk voice dictation"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.parse_args()

    print(f"SoupaWhisper v{__version__}")
    print(f"Config: {CONFIG_PATH}")

    check_dependencies()

    dictation = Dictation()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
