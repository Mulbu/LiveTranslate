"""
LiveTranslate - Linux audio capture via PulseAudio (pulsectl + parec).
Replaces the original WASAPI-based audio_capture.py for Linux compatibility.

Captures system audio via PulseAudio monitor source (loopback)
and optional microphone input.
"""

import logging
import threading
import queue
import time
import subprocess
import numpy as np
import pulsectl

log = logging.getLogger("LiveTranslate.Audio")

DEVICE_CHECK_INTERVAL = 2.0  # seconds

# ── PulseAudio device helpers ──────────────────────────────────────────────


def _get_pulse():
    """Create a temporary PulseAudio connection for enumeration."""
    return pulsectl.Pulse("livetranslate-enum")


def _monitor_name_for_sink(pulse, sink_name):
    """Return the monitor source name for a given sink name."""
    for src in pulse.source_list():
        if src.monitor_of_sink_name == sink_name:
            return src.name
    return None


def list_output_devices():
    """Return list of PulseAudio sink (output) descriptions (for loopback capture)."""
    try:
        pulse = _get_pulse()
        devices = []
        for sink in pulse.sink_list():
            devices.append(sink.description)
        pulse.close()
        return devices
    except Exception as e:
        log.warning(f"Failed to list output devices: {e}")
        return []


def list_input_devices():
    """Return list of PulseAudio source (input/mic) descriptions."""
    try:
        pulse = _get_pulse()
        devices = []
        for src in pulse.source_list():
            # Exclude monitor sources (these are loopback captures of sinks)
            if not src.monitor_of_sink_name:
                devices.append(src.description)
        pulse.close()
        return devices
    except Exception as e:
        log.warning(f"Failed to list input devices: {e}")
        return []


def _description_to_name(pulse, desc, kind="sink"):
    """Convert a human-readable description back to a PulseAudio name."""
    if kind == "sink":
        for s in pulse.sink_list():
            if s.description == desc:
                return s.name
    else:
        for s in pulse.source_list():
            if s.description == desc:
                return s.name
    return None


# ── AudioCapture class ─────────────────────────────────────────────────────


class AudioCapture:
    """Capture system audio via PulseAudio monitor + optional mic via parec."""

    def __init__(self, device=None, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_queue = queue.Queue(maxsize=100)
        self._device_name = device  # None = default sink, "__disabled__" = off
        self._current_device_name = None
        self._loopback_disabled = False
        self._running = False
        self._read_thread = None
        self._loopback_proc = None  # subprocess for parec (loopback)
        self._mic_proc = None  # subprocess for parec (mic)
        # Microphone
        self._mic_device_name = None
        self._mic_buf = np.array([], dtype=np.float32)
        # Locks and events
        self._lock = threading.Lock()
        self._restart_event = threading.Event()
        self._mic_restart_event = threading.Event()

    # ── Public API ──────────────────────────────────────────────────────

    def set_device(self, device_name):
        """Change capture device at runtime. None = system default, '__disabled__' = off."""
        if device_name == self._device_name:
            return
        log.info(f"Audio device changed: {self._device_name} -> {device_name}")
        self._device_name = device_name
        self._loopback_disabled = device_name == "__disabled__"
        if self._running:
            self._restart_event.set()

    def set_mic_device(self, device_name):
        """Change mic device at runtime. None = disabled."""
        if device_name == self._mic_device_name:
            return
        log.info(f"Mic device changed: {self._mic_device_name} -> {device_name}")
        self._mic_device_name = device_name
        if self._running:
            self._mic_restart_event.set()

    def start(self):
        self._loopback_disabled = self._device_name == "__disabled__"
        self._open_capture_streams()
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        log.info("Audio capture started")

    def stop(self):
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=3)
        self._close_capture_streams()
        log.info("Audio capture stopped")

    def get_audio(self, timeout=1.0):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal: stream management ─────────────────────────────────────

    def _get_loopback_source_name(self):
        """Resolve the PulseAudio monitor source name for the target sink."""
        pulse = pulsectl.Pulse("livetranslate-loopback")
        try:
            # Determine target sink name
            if self._device_name is None:
                # Use default sink
                sink_info = pulse.get_sink_by_name(
                    pulse.server_info().default_sink_name
                )
                target_sink_name = sink_info.name
            else:
                # Find sink by description
                target_sink_name = _description_to_name(
                    pulse, self._device_name, "sink"
                )
                if target_sink_name is None:
                    log.warning(f"Sink not found: {self._device_name}, using default")
                    sink_info = pulse.get_sink_by_name(
                        pulse.server_info().default_sink_name
                    )
                    target_sink_name = sink_info.name

            # Get monitor source for that sink
            monitor_name = _monitor_name_for_sink(pulse, target_sink_name)
            if monitor_name is None:
                raise RuntimeError(
                    f"No monitor source found for sink {target_sink_name}"
                )

            # Get sink description for logging
            sink = pulse.get_sink_by_name(target_sink_name)
            self._current_device_name = sink.description

            log.info(
                f"Loopback source: {monitor_name} (sink: {target_sink_name}, desc: {sink.description})"
            )
            return monitor_name
        finally:
            pulse.close()

    def _get_mic_source_name(self):
        """Resolve the PulseAudio source name for the mic."""
        if not self._mic_device_name:
            return None
        if self._mic_device_name in ("__default__", "default"):
            pulse = pulsectl.Pulse("livetranslate-mic")
            try:
                return pulse.server_info().default_source_name
            finally:
                pulse.close()

        pulse = pulsectl.Pulse("livetranslate-mic")
        try:
            src_name = _description_to_name(
                pulse, self._mic_device_name, "source"
            )
            if src_name is None:
                log.warning(f"Mic source not found: {self._mic_device_name}")
            return src_name
        finally:
            pulse.close()

    def _open_loopback_stream(self):
        """Start parec subprocess for loopback (monitor) capture."""
        source_name = self._get_loopback_source_name()
        if source_name is None:
            raise RuntimeError("Could not determine loopback source")

        chunk_size = int(self.sample_rate * self.chunk_duration)
        cmd = [
            "parec",
            "--format=float32",
            "--channels=1",
            f"--rate={self.sample_rate}",
            f"--latency-msec={int(self.chunk_duration * 1000)}",
            f"--device={source_name}",
            "--raw",
        ]
        log.debug(f"Starting parec (loopback): {' '.join(cmd)}")
        self._loopback_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=chunk_size * 4,  # float32 = 4 bytes
        )

    def _open_mic_stream(self):
        """Start parec subprocess for microphone capture."""
        source_name = self._get_mic_source_name()
        if source_name is None:
            log.warning("No mic source resolved, mic disabled")
            return

        chunk_size = int(self.sample_rate * self.chunk_duration)
        cmd = [
            "parec",
            "--format=float32",
            "--channels=1",
            f"--rate={self.sample_rate}",
            f"--latency-msec={int(self.chunk_duration * 1000)}",
            f"--device={source_name}",
            "--raw",
        ]
        log.debug(f"Starting parec (mic): {' '.join(cmd)}")
        self._mic_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=chunk_size * 4,
        )

    def _close_loopback_stream(self):
        if self._loopback_proc:
            try:
                self._loopback_proc.terminate()
                self._loopback_proc.wait(timeout=2)
            except Exception:
                try:
                    self._loopback_proc.kill()
                except Exception:
                    pass
            self._loopback_proc = None

    def _close_mic_stream(self):
        if self._mic_proc:
            try:
                self._mic_proc.terminate()
                self._mic_proc.wait(timeout=2)
            except Exception:
                try:
                    self._mic_proc.kill()
                except Exception:
                    pass
            self._mic_proc = None

    def _open_capture_streams(self):
        if not self._loopback_disabled:
            try:
                self._open_loopback_stream()
            except Exception as e:
                log.error(f"Failed to open loopback stream: {e}")
        else:
            log.info("Loopback disabled (mic-only mode)")

        if self._mic_device_name:
            try:
                self._open_mic_stream()
            except Exception as e:
                log.warning(f"Failed to open mic stream: {e}")

    def _close_capture_streams(self):
        self._close_loopback_stream()
        self._close_mic_stream()

    def _restart_streams(self):
        """Restart all capture streams."""
        with self._lock:
            self._close_capture_streams()
            if not self._loopback_disabled:
                self._open_loopback_stream()
            else:
                log.info("Loopback disabled (mic-only mode)")
            if self._mic_device_name:
                self._close_mic_stream()
                try:
                    self._open_mic_stream()
                except Exception as e:
                    log.warning(f"Failed to re-open mic after restart: {e}")

    def _read_parec_float32(self, proc, n_bytes):
        """Read exactly n_bytes from parec stdout as float32 array."""
        data = b""
        while len(data) < n_bytes:
            chunk = proc.stdout.read(n_bytes - len(data))
            if not chunk:
                # EOF — parec probably exited
                return None
            data += chunk
        return np.frombuffer(data, dtype=np.float32)

    # ── Read loop ───────────────────────────────────────────────────────

    def _read_loop(self):
        """Background thread: read audio chunks and push to queue."""
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        chunk_bytes = chunk_samples * 4  # float32 = 4 bytes

        last_device_check = time.monotonic()

        while self._running:
            # Handle pending restart request
            if self._restart_event.is_set():
                self._restart_event.clear()
                try:
                    self._restart_streams()
                    # Drain old audio
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    log.info(
                        f"Audio capture restarted on: {self._current_device_name}"
                    )
                except Exception as e:
                    log.error(f"Restart after device change failed: {e}")
                    time.sleep(0.5)
                continue

            # Handle mic device change
            if self._mic_restart_event.is_set():
                self._mic_restart_event.clear()
                self._close_mic_stream()
                self._mic_buf = np.array([], dtype=np.float32)
                if self._mic_device_name:
                    try:
                        self._open_mic_stream()
                        log.info(f"Mic stream opened: {self._mic_device_name}")
                    except Exception as e:
                        log.error(f"Failed to open mic: {e}")
                else:
                    log.info("Mic disabled")

            # Read loopback chunk
            loopback_audio = None
            if self._loopback_disabled:
                time.sleep(self.chunk_duration)
                loopback_audio = np.zeros(chunk_samples, dtype=np.float32)
            elif self._loopback_proc:
                with self._lock:
                    if self._loopback_proc and self._loopback_proc.stdout:
                        try:
                            audio = self._read_parec_float32(
                                self._loopback_proc, chunk_bytes
                            )
                            if audio is not None and len(audio) == chunk_samples:
                                loopback_audio = audio
                            elif audio is None:
                                log.warning("parec (loopback) stream ended")
                                time.sleep(0.5)
                                self._restart_streams()
                                continue
                        except Exception as e:
                            log.warning(f"Loopback read error: {e}")
                            time.sleep(0.1)
                            continue

            # Read mic data
            if self._mic_proc and self._mic_proc.stdout:
                try:
                    # Read all available mic data (non-blocking-ish: read what's there)
                    import select
                    import os

                    fd = self._mic_proc.stdout.fileno()
                    available = 0
                    try:
                        # Check how much is readable
                        _, _, _ = select.select([fd], [], [], 0)
                        import fcntl

                        available = fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ, 0)
                        available = min(available, 65536)  # cap at 64KB
                    except Exception:
                        available = chunk_bytes

                    if available >= chunk_bytes:
                        mic_audio = self._read_parec_float32(
                            self._mic_proc, available
                        )
                        if mic_audio is not None:
                            self._mic_buf = np.concatenate([self._mic_buf, mic_audio])
                except Exception as e:
                    log.warning(f"Mic read error: {e}")

            if loopback_audio is None:
                time.sleep(0.005)
                continue

            # Mix: take matching length from mic buffer
            audio = loopback_audio
            mic_rms = None
            if len(self._mic_buf) > 0:
                n = len(loopback_audio)
                if len(self._mic_buf) >= n:
                    mic_chunk = self._mic_buf[:n]
                    self._mic_buf = self._mic_buf[n:]
                else:
                    mic_chunk = np.zeros(n, dtype=np.float32)
                    mic_chunk[: len(self._mic_buf)] = self._mic_buf
                    self._mic_buf = np.array([], dtype=np.float32)
                mic_rms = float(np.sqrt(np.mean(mic_chunk**2)))
                audio = loopback_audio + mic_chunk

            try:
                self.audio_queue.put_nowait((audio, mic_rms))
            except queue.Full:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait((audio, mic_rms))

            # Periodic device check (to detect default output changes)
            now = time.monotonic()
            if now - last_device_check > DEVICE_CHECK_INTERVAL:
                last_device_check = now
                if self._device_name is None:
                    try:
                        pulse = pulsectl.Pulse("livetranslate-check")
                        try:
                            current_default = pulse.server_info().default_sink_name
                            if self._current_device_name:
                                # Compare descriptions
                                sink = pulse.get_sink_by_name(current_default)
                                if sink.description != self._current_device_name:
                                    log.info(
                                        f"Default output changed: "
                                        f"{self._current_device_name} -> {sink.description}"
                                    )
                                    log.info("Restarting audio capture...")
                                    self._restart_streams()
                                    log.info(
                                        f"Audio capture restarted on: {self._current_device_name}"
                                    )
                        finally:
                            pulse.close()
                    except Exception as e:
                        log.warning(f"Device check error: {e}")

    def __del__(self):
        self._close_capture_streams()
