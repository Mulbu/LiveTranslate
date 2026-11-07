"""Punctuation restoration post-processor.

Mirrors the ``Punctuator`` used by ``kotoba-tech/kotoba-whisper-v2.2``
(pipeline/kotoba_whisper.py): the ONNX ``punctuators`` model restores
punctuation / true-casing / sentence boundaries. The same model id and the
same "skip if already punctuated" guard are used so behaviour matches v2.2.

The model is ONNX (onnxruntime) and runs fine on CPU, so it is cheap enough to
keep inside the ASR worker next to the transcriber.
"""

from __future__ import annotations

import logging

log = logging.getLogger("LiveTranslate.Punctuation")

DEFAULT_MODEL = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"

# Languages where sentence-final tokens are CJK punctuation -> join without spaces.
_CJK_LANGS = {"ja", "zh", "yue", "ko"}


class PunctuationRestorer:
    """Restore punctuation / casing / sentence segmentation for ASR text."""

    # kotoba guard: if any of these already appear, assume punctuated.
    _EXISTING = ("!", "?", "、", "。", "！", "？")

    def __init__(self, model: str = DEFAULT_MODEL):
        from punctuators.models import PunctCapSegModelONNX

        self._model = PunctCapSegModelONNX.from_pretrained(model or DEFAULT_MODEL)
        log.info(f"Punctuation model loaded: {model or DEFAULT_MODEL}")

    def punctuate(self, text: str, language: str | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return text

        # Skip if the engine already produced punctuation.
        if any(p in text for p in self._EXISTING):
            return text

        try:
            sentences = self._model.infer([text])[0]
        except Exception as e:  # pragma: no cover - defensive
            log.warning(f"Punctuation inference failed: {e}")
            return text

        if not sentences:
            return text

        sep = "" if (language in _CJK_LANGS) else " "
        punctuated = sep.join(s.strip() for s in sentences).strip()

        # kotoba's guard: model emits literal "unk" on failure.
        if not punctuated or "unk" in punctuated.lower():
            return text
        return punctuated
