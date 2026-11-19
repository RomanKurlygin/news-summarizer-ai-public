"""Посэмпловые метрики с нормализацией и макроусреднением."""

import re
import threading
import unicodedata
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel

BERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BERT_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
BERT_LAYERS = 12
_WORDS = re.compile(r"[^\W_]+")


class MetricsUnavailable(RuntimeError):
    """Недоступна библиотека метрик, её словарь или модель."""


class SummaryMetrics(BaseModel):
    bleu_1: float
    bleu_2: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    meteor: float
    chrf: float
    chrf_plus_plus: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    exact_match: float


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


class _UnicodeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return _WORDS.findall(text.casefold())


def average_metrics(items: list[SummaryMetrics]) -> SummaryMetrics:
    return SummaryMetrics.model_validate({
        key: fmean(getattr(item, key) for item in items) for key in SummaryMetrics.model_fields
    })


class MetricsCalculator:
    def __init__(self, cache_dir: Path, nltk_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.nltk_dir = nltk_dir
        self._scorer = None
        self._lock = threading.Lock()

    def metadata(self) -> dict:
        return {
            "libraries": {name: version(name) for name in ("sacrebleu", "rouge-score", "nltk", "bert-score")},
            "bertscore_model": BERT_MODEL, "bertscore_revision": BERT_REVISION,
            "bertscore_layers": BERT_LAYERS, "averaging": "macro_per_example",
        }

    def _semantic(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        from bert_score import BERTScorer
        from huggingface_hub import snapshot_download

        if self._scorer is None:
            snapshot = snapshot_download(BERT_MODEL, revision=BERT_REVISION, cache_dir=self.cache_dir,
                                         local_files_only=True)
            self._scorer = BERTScorer(model_type=snapshot, num_layers=BERT_LAYERS, device="cpu",
                                     use_fast_tokenizer=True, batch_size=8)
        precision, recall, f1 = self._scorer.score([hyp for hyp, _ in pairs], [ref for _, ref in pairs])
        return list(zip(precision.tolist(), recall.tolist(), f1.tolist(), strict=True))

    def calculate(self, pairs: list[tuple[str, str]]) -> list[SummaryMetrics]:
        """Сравнить пары (гипотеза, эталон) в исходном порядке."""
        if not pairs:
            raise ValueError("Для метрик нужна хотя бы одна пара.")
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            from rouge_score.rouge_scorer import RougeScorer
            from sacrebleu.metrics import BLEU, CHRF

            directory = str(self.nltk_dir)
            if directory not in nltk.data.path:
                nltk.data.path.append(directory)
            tokenizer = _UnicodeTokenizer()
            rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=tokenizer)
            bleu = [BLEU(max_ngram_order=order, tokenize="13a", lowercase=True,
                         smooth_method="exp", effective_order=True) for order in (1, 2)]
            chrf = [CHRF(char_order=6, word_order=order, beta=2) for order in (0, 2)]
            normalized = [(_normalize(hyp), _normalize(ref)) for hyp, ref in pairs]
            nonempty = [(hyp, ref) for hyp, ref in normalized if hyp and ref]
            if nonempty:
                with self._lock:
                    semantic = iter(self._semantic(nonempty))
            result = []
            for hyp, ref in normalized:
                if not hyp or not ref:
                    result.append(SummaryMetrics.model_validate({key: 0.0 for key in SummaryMetrics.model_fields}))
                    continue
                rouge_scores = rouge.score(ref, hyp)
                precision, recall, f1 = next(semantic)
                result.append(SummaryMetrics(
                    bleu_1=min(1.0, bleu[0].sentence_score(hyp, [ref]).score / 100),
                    bleu_2=min(1.0, bleu[1].sentence_score(hyp, [ref]).score / 100),
                    rouge_1=rouge_scores["rouge1"].fmeasure, rouge_2=rouge_scores["rouge2"].fmeasure,
                    rouge_l=rouge_scores["rougeL"].fmeasure,
                    meteor=meteor_score([tokenizer.tokenize(ref)], tokenizer.tokenize(hyp)),
                    chrf=chrf[0].sentence_score(hyp, [ref]).score / 100,
                    chrf_plus_plus=chrf[1].sentence_score(hyp, [ref]).score / 100,
                    bertscore_precision=precision, bertscore_recall=recall, bertscore_f1=f1,
                    exact_match=float(hyp.casefold() == ref.casefold()),
                ))
            return result
        except (ImportError, LookupError, OSError, RuntimeError, ValueError) as exc:
            raise MetricsUnavailable("Проверьте библиотеки оценки, WordNet и локальный снимок модели BERTScore.") from exc
