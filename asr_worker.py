import gc
import inspect
import logging
import sys
import traceback
from typing import Any

import numpy as np

log = logging.getLogger("LiveTranslate.ASRWorker")


def _setup_logging():
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)


def _error_response(msg_id: str | None, exc: BaseException, recoverable: bool) -> dict:
    return {
        "id": msg_id,
        "ok": False,
        "type": "error",
        "error": {
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "recoverable": recoverable,
        },
    }


def _ok_response(msg_id: str | None, response_type: str, payload: Any = None) -> dict:
    return {
        "id": msg_id,
        "ok": True,
        "type": response_type,
        "payload": payload,
    }


def _parse_device(device: str) -> tuple[str, int]:
    device = str(device or "cpu").split(" (", 1)[0].strip()
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        return "cuda", index
    return device, 0


def _load_engine(config: dict):
    from model_manager import MODELS_DIR, apply_cache_env

    apply_cache_env()

    engine_type = config["engine_type"]
    device = config.get("device", "cpu")
    hub = config.get("hub", "ms")
    language = config.get("language", "auto")
    pad_seconds = config.get("pad_seconds")

    parsed_device, device_index = _parse_device(device)

    if engine_type == "funasr":
        from asr_funasr import FunASREngine

        engine = FunASREngine(
            model_key=config.get("funasr_model"),
            device=device,
            hub=hub,
            pad_seconds=pad_seconds,
        )
    elif engine_type == "anime-whisper":
        from asr_anime_whisper import AnimeWhisperEngine

        worker_device = parsed_device if parsed_device == "cpu" else f"cuda:{device_index}"
        engine = AnimeWhisperEngine(device=worker_device, hub=hub)
    else:
        from asr_engine import ASREngine

        compute_type = config.get("compute_type", "float16")
        if parsed_device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        download_root = config.get("download_root")
        if not download_root:
            download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
        engine = ASREngine(
            model_size=config["model_size"],
            device=parsed_device,
            device_index=device_index,
            compute_type=compute_type,
            language=language,
            download_root=download_root,
            pad_seconds=pad_seconds,
        )

    if hasattr(engine, "set_language"):
        engine.set_language(language)
    return engine


def _transcribe(engine, payload: dict):
    audio = payload.get("audio")
    if not isinstance(audio, np.ndarray):
        raise TypeError("transcribe payload audio must be a numpy.ndarray")

    kwargs = {}
    signature = inspect.signature(engine.transcribe)
    if "word_timestamps" in signature.parameters:
        kwargs["word_timestamps"] = bool(payload.get("word_timestamps", False))
    return engine.transcribe(audio, **kwargs)


def _cleanup_engine(engine):
    if engine is not None and hasattr(engine, "unload"):
        try:
            engine.unload()
        except Exception:
            log.warning("ASR engine unload failed", exc_info=True)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _PostProcessor:
    """Runs optional punctuation restoration + online speaker ID on ASR results.

    Built once per worker process (models stay resident), applied per committed
    segment. Both components degrade to no-ops if their deps/models are missing.
    """

    def __init__(self, restorer=None, speaker=None):
        self._restorer = restorer
        self._speaker = speaker

    def apply(self, result, audio):
        if not isinstance(result, dict):
            return result
        if self._restorer is not None:
            try:
                result["text"] = self._restorer.punctuate(
                    result.get("text", ""), result.get("language")
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(f"Punctuation post-process failed: {exc}")
        if self._speaker is not None:
            try:
                speaker = self._speaker.identify(audio)
                if speaker:
                    result["speaker"] = speaker
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(f"Speaker ID post-process failed: {exc}")
        return result


def _build_postprocessor(config: dict) -> "_PostProcessor | None":
    restorer = None
    speaker = None

    if config.get("add_punctuation"):
        try:
            from punctuation import PunctuationRestorer

            restorer = PunctuationRestorer(model=config.get("punctuation_model"))
        except Exception as exc:
            log.warning(f"Punctuation disabled (init failed): {exc}", exc_info=True)

    if config.get("diarization"):
        try:
            from speaker_id import SpeakerIdentifier

            parsed_device, device_index = _parse_device(config.get("device", "cpu"))
            worker_device = (
                parsed_device if parsed_device == "cpu" else f"cuda:{device_index}"
            )
            speaker = SpeakerIdentifier(
                model=config.get("diarization_model"),
                device=worker_device,
                threshold=float(config.get("diarization_threshold", 0.72)),
                max_speakers=int(config.get("diarization_max_speakers", 8)),
                min_segment=float(config.get("diarization_min_segment", 0.4)),
            )
        except Exception as exc:
            log.warning(f"Speaker diarization disabled (init failed): {exc}", exc_info=True)

    if restorer is None and speaker is None:
        return None
    return _PostProcessor(restorer=restorer, speaker=speaker)


def _warmup_engine(engine, config: dict):
    """One tiny inference at startup to absorb CUDA/backend init cost.

    The model is already loaded when this runs, but ctranslate2/CUDA lazily
    initialise kernels on the first real ``transcribe`` -> the very first
    utterance otherwise pays a one-off ~5-7s. A 1s silence pass avoids that.
    """
    if config.get("engine_type") not in ("whisper", "anime-whisper"):
        return
    if not hasattr(engine, "transcribe"):
        return
    try:
        import numpy as np

        engine.transcribe(np.zeros(16000, dtype=np.float32))
        log.info("ASR warmup complete")
    except Exception as exc:  # pragma: no cover - best effort only
        log.warning(f"ASR warmup skipped: {exc}")


def worker_main(conn, config: dict):
    _setup_logging()
    engine = None
    post = None
    try:
        log.info(
            "ASR worker loading: "
            f"{config.get('engine_type')} on {config.get('device')} "
            f"(pid config={config.get('display_name', '')})"
        )
        engine = _load_engine(config)
        try:
            post = _build_postprocessor(config)
        except BaseException as exc:  # never let post-processing block ASR
            log.warning(f"Post-processor init failed, disabled: {exc}", exc_info=True)
            post = None
        conn.send(
            _ok_response(
                None,
                "ready",
                {
                    "engine_type": config.get("engine_type"),
                    "display_name": config.get("display_name"),
                    "device": config.get("device"),
                },
            )
        )
    except BaseException as exc:
        log.error(f"ASR worker load failed: {exc}", exc_info=True)
        try:
            conn.send(_error_response(None, exc, recoverable=False))
        finally:
            _cleanup_engine(engine)
            conn.close()
        return

    _warmup_engine(engine, config)

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            msg_id = msg.get("id")
            msg_type = msg.get("type")
            payload = msg.get("payload") or {}

            try:
                if msg_type == "shutdown":
                    conn.send(_ok_response(msg_id, "shutdown"))
                    break
                if msg_type == "transcribe":
                    result = _transcribe(engine, payload)
                    if post is not None and payload.get("postprocess", True):
                        result = post.apply(result, payload.get("audio"))
                    conn.send(_ok_response(msg_id, "result", result))
                    continue
                if msg_type == "set_language":
                    if hasattr(engine, "set_language"):
                        engine.set_language(payload.get("language", "auto"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                if msg_type == "set_input_padding":
                    if hasattr(engine, "set_input_padding"):
                        engine.set_input_padding(payload.get("pad_seconds"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                if msg_type == "set_postprocess":
                    # Rebuild ONLY the punctuation / speaker post-processor while
                    # keeping the (expensive) ASR model loaded in memory.
                    try:
                        changes = payload.get("config") or {}
                        new_cfg = dict(config)
                        new_cfg.update(changes)
                        post = _build_postprocessor(new_cfg)
                        config.update(changes)
                        conn.send(_ok_response(msg_id, "ack"))
                    except Exception as exc:
                        log.error(f"set_postprocess failed: {exc}", exc_info=True)
                        conn.send(_error_response(msg_id, exc, recoverable=True))
                    continue
                raise ValueError(f"Unknown ASR worker command: {msg_type}")
            except Exception as exc:
                log.error(f"ASR worker command failed: {msg_type}: {exc}", exc_info=True)
                conn.send(_error_response(msg_id, exc, recoverable=True))
    finally:
        _cleanup_engine(engine)
        try:
            conn.close()
        except Exception:
            pass
        log.info("ASR worker stopped")
