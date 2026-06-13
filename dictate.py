#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper.
Hold the hotkey to record, release to transcribe and copy to clipboard.
"""

import argparse
import configparser
from concurrent.futures import ThreadPoolExecutor
import json
import platform
import re
import subprocess
import tempfile
import threading
import signal
import sys
import os
from pathlib import Path

from pynput import keyboard

__version__ = "0.1.0"

IS_MACOS = platform.system() == "Darwin"
# On Linux the clipboard/typing backend and hotkey strategy differ between X11
# (xclip/xdotool, global key listener) and Wayland (wl-copy/wtype, no global
# grab — the compositor binds a key to `dictate.py toggle` instead).
IS_WAYLAND = not IS_MACOS and os.environ.get("XDG_SESSION_TYPE") == "wayland"

if IS_MACOS:
    # Metal-accelerated Whisper on Apple Silicon
    import mlx_whisper
else:
    from faster_whisper import WhisperModel

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"
# Daemon mode (Wayland) writes its PID here so `dictate.py toggle` can signal it
PIDFILE = Path(tempfile.gettempdir()) / "soupawhisper.pid"


def load_config():
    config = configparser.ConfigParser()

    # Defaults
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "language": "",
        "vocabulary": "",
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
        "vocabulary": config.get("whisper", "vocabulary", fallback=defaults["vocabulary"]),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
        "replacements": dict(config.items("replacements")) if config.has_section("replacements") else {},
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


def get_mlx_model_repo(name):
    """Full HF repo paths pass through; bare sizes map to mlx-community repos."""
    return name if "/" in name else f"mlx-community/whisper-{name}-mlx"


def load_wav(path):
    """Load a 16kHz mono 16-bit WAV as float32 in [-1, 1].

    mlx-whisper shells out to ffmpeg when given a path; our recordings are
    already in Whisper's native format, so decode them without it.
    """
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


HOTKEY = get_hotkey(CONFIG["key"])
# macOS virtual keycode of the hotkey; None for character keys (unknown keycode)
HOTKEY_VK = HOTKEY.value.vk if isinstance(HOTKEY, keyboard.Key) else HOTKEY.vk
MODEL_SIZE = CONFIG["model"]
MODEL_REPO = get_mlx_model_repo(MODEL_SIZE) if IS_MACOS else None
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
LANGUAGE = CONFIG["language"] or None  # None = auto-detect
# Domain terms fed to Whisper as initial_prompt to bias their spelling
INITIAL_PROMPT = CONFIG["vocabulary"].strip() or None
REPLACEMENTS = CONFIG["replacements"]


def apply_replacements(text):
    """Fix recurring mistranscriptions the model cannot get right.

    Matches case-insensitively from a word boundary but replaces only the
    matched prefix, so inflected forms keep their suffix
    ("Heilangiem" -> "ChiLangiem").
    """
    for wrong, right in REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(wrong)}", right, text, flags=re.IGNORECASE)
    return text
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

        if IS_MACOS:
            # MLX streams are bound to the thread that created them, so a
            # single thread must own all MLX work (model load + transcribe)
            self.mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

        # Load model in background
        print(f"Loading Whisper model ({MODEL_SIZE})...")
        threading.Thread(target=self._load_model, daemon=True).start()

        # system_profiler takes a couple of seconds, don't block startup
        threading.Thread(target=print_input_device, daemon=True).start()

    def _load_model(self):
        try:
            if IS_MACOS:
                # Download and load the model upfront; ModelHolder caches it
                # for the mlx_whisper.transcribe() calls
                def load():
                    import mlx.core as mx
                    from mlx_whisper.transcribe import ModelHolder
                    ModelHolder.get_model(MODEL_REPO, mx.float16)

                self.mlx.submit(load).result()
            else:
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
            if IS_MACOS:
                result = self.mlx.submit(
                    mlx_whisper.transcribe,
                    load_wav(self.temp_file.name),
                    path_or_hf_repo=MODEL_REPO,
                    language=LANGUAGE,
                    initial_prompt=INITIAL_PROMPT,
                ).result()
                text = result["text"].strip()
            else:
                segments, info = self.model.transcribe(
                    self.temp_file.name,
                    beam_size=5,
                    vad_filter=True,
                    language=LANGUAGE,
                    initial_prompt=INITIAL_PROMPT,
                )
                text = " ".join(segment.text.strip() for segment in segments)

            text = apply_replacements(text)

            if text:
                # Copy to clipboard
                if IS_MACOS:
                    clipboard_cmd = ["pbcopy"]
                elif IS_WAYLAND:
                    clipboard_cmd = ["wl-copy"]
                else:
                    clipboard_cmd = ["xclip", "-selection", "clipboard"]
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
                    elif IS_WAYLAND:
                        subprocess.run(["wtype", text])
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

    def toggle(self):
        """Toggle recording on/off (driven by an external signal)."""
        self._spawn(self.stop_recording if self.recording else self.start_recording)

    def run_daemon(self):
        """Run headless, toggling on SIGUSR1.

        Used on Wayland, where pynput cannot grab a global hotkey: the
        compositor binds a key to `dictate.py toggle`, which signals us.
        """
        PIDFILE.write_text(str(os.getpid()))
        signal.signal(signal.SIGUSR1, lambda sig, frame: self.toggle())
        print(f"Daemon mode. Send SIGUSR1 (kill -USR1 {os.getpid()}) or run: dictate.py toggle")
        try:
            while self.running:
                signal.pause()
        finally:
            PIDFILE.unlink(missing_ok=True)

    def stop(self):
        print("\nExiting...")
        self.running = False
        PIDFILE.unlink(missing_ok=True)
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
    elif IS_WAYLAND:
        required = [("arecord", "alsa-utils"), ("wl-copy", "wl-clipboard")]
        if AUTO_TYPE:
            required.append(("wtype", "wtype"))
        install_hint = "sudo apt install"
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


def send_toggle():
    """Signal a running daemon to start/stop recording, then exit."""
    if not PIDFILE.exists():
        print("SoupaWhisper daemon is not running.")
        sys.exit(1)
    try:
        os.kill(int(PIDFILE.read_text().strip()), signal.SIGUSR1)
    except (ProcessLookupError, ValueError):
        print("SoupaWhisper daemon is not running (stale pidfile).")
        PIDFILE.unlink(missing_ok=True)
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
    parser.add_argument(
        "command",
        nargs="?",
        choices=["toggle", "daemon"],
        help="toggle: signal a running daemon to start/stop recording. "
             "daemon: run headless, controlled via SIGUSR1 (used on Wayland)."
    )
    args = parser.parse_args()

    if args.command == "toggle":
        send_toggle()
        return

    print(f"SoupaWhisper v{__version__}")
    print(f"Config: {CONFIG_PATH}")

    check_dependencies()

    dictation = Dictation()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    # Wayland has no global hotkey grab, so fall back to signal-driven daemon
    # mode automatically; X11 and macOS use the push-to-talk key listener.
    if args.command == "daemon" or IS_WAYLAND:
        dictation.run_daemon()
    else:
        dictation.run()


if __name__ == "__main__":
    main()
