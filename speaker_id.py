"""Online speaker identification post-processor.

``kotoba-whisper-v2.2`` performs *offline* speaker diarization over a whole
file with ``pyannote/speaker-diarization-3.1``. This app streams 1-8s VAD
segments, where whole-file diarization cannot give consistent labels across
segments. Instead we use a pyannote *speaker embedding* model and cluster
embeddings online: each new segment is matched to the nearest known speaker
(cosine >= threshold) or registered as a new speaker (up to ``max_speakers``).

This yields stable ``SPEAKER_00`` / ``SPEAKER_01`` ... labels across a session
while staying fast enough for real-time use.

Requires ``pyannote.audio`` and a gated embedding model (see
POSTPROCESS_SETUP.md). All failures degrade gracefully (returns ``None`` and the
subtitle simply has no speaker tag).
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("LiveTranslate.SpeakerID")

DEFAULT_MODEL = "pyannote/embedding"


class SpeakerIdentifier:
    """Assign a stable speaker label to each audio segment via embedding clustering."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cpu",
        threshold: float = 0.72,
        max_speakers: int = 8,
        min_segment: float = 0.4,
    ):
        import torch
        from pyannote.audio import Inference, Model

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA unavailable for speaker ID, falling back to CPU")
            device = "cpu"

        self._torch = torch
        self._device = torch.device(device)
        self._threshold = float(threshold)
        self._max_speakers = max(1, int(max_speakers))
        self._min_segment = float(min_segment)

        emb_model = Model.from_pretrained(model or DEFAULT_MODEL)
        self._inference = Inference(emb_model, window="whole")
        self._inference.to(self._device)

        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._lock = threading.Lock()

        log.info(
            f"Speaker ID ready: model={model or DEFAULT_MODEL} device={device} "
            f"threshold={self._threshold} max_speakers={self._max_speakers}"
        )

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        wav = self._torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        emb = self._inference(
            {"waveform": wav.to(self._device), "sample_rate": sample_rate}
        )
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm <= 0.0:
            return None
        return emb / norm

    @staticmethod
    def _label(index: int) -> str:
        return f"SPEAKER_{index:02d}"

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> str | None:
        if audio is None:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0 or (audio.size / float(sample_rate)) < self._min_segment:
            return None

        with self._lock:
            try:
                emb = self._embed(audio, sample_rate)
            except Exception as e:  # pragma: no cover - defensive
                log.warning(f"Speaker embedding failed: {e}")
                return None
            if emb is None:
                return None

            if not self._centroids:
                self._centroids.append(emb)
                self._counts.append(1)
                return self._label(0)

            sims = [float(np.dot(emb, c)) for c in self._centroids]
            best = int(np.argmax(sims))

            if sims[best] >= self._threshold:
                # Running-average centroid update keeps it adaptive.
                n = self._counts[best]
                merged = (self._centroids[best] * n + emb) / (n + 1)
                norm = float(np.linalg.norm(merged))
                self._centroids[best] = merged / norm if norm > 0 else merged
                self._counts[best] = n + 1
                return self._label(best)

            if len(self._centroids) < self._max_speakers:
                self._centroids.append(emb)
                self._counts.append(1)
                return self._label(len(self._centroids) - 1)

            # Reached the speaker cap: fall back to the closest existing speaker.
            return self._label(best)

    def reset(self) -> None:
        with self._lock:
            self._centroids.clear()
            self._counts.clear()
