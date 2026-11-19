import os

import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("rouge_score")
pytest.importorskip("nltk")

from backend.core.config import Settings
from backend.evaluation.metrics_calculator import BERT_MODEL, MetricsCalculator, average_metrics


def test_lexical_metrics_and_order(tmp_path, monkeypatch):
    calculator = MetricsCalculator(tmp_path, tmp_path)
    seen = []

    def semantic(pairs):
        seen.extend(pairs)
        return [(1.0, 1.0, 1.0), (0.2, 0.3, 0.24)]

    monkeypatch.setattr(calculator, "_semantic", semantic)
    perfect, empty, single = calculator.calculate([
        ("  THE   SAME WORDS ", "the same words"),
        (" ", "empty"),
        ("rain", "rain"),
    ])
    # chrF и BERTScore сохраняют регистр, остальные метрики здесь используют нормализованные слова.
    for key in ("bleu_1", "bleu_2", "rouge_1", "rouge_2", "rouge_l", "exact_match"):
        assert getattr(perfect, key) == pytest.approx(1)
    assert perfect.meteor == pytest.approx(1 - 0.5 * (1 / 3) ** 3)
    assert perfect.chrf < 1 and perfect.chrf_plus_plus < 1
    assert all(number == 0 for number in empty.model_dump().values())
    assert seen == [("THE SAME WORDS", "the same words"), ("rain", "rain")]
    assert single.bleu_1 == single.bleu_2 == single.chrf == single.chrf_plus_plus == 1
    assert single.rouge_2 == 0 and single.rouge_l == 1
    assert single.bertscore_precision == 0.2 and single.bertscore_recall == 0.3
    assert average_metrics([perfect, empty, single]).bertscore_f1 == pytest.approx((1 + 0.24) / 3)


def test_empty_pair_collection_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="хотя бы"):
        MetricsCalculator(tmp_path, tmp_path).calculate([])


@pytest.mark.skipif(os.getenv("RUN_LIVE_EMBEDDINGS") != "1", reason="Требуется локальная модель MiniLM")
def test_real_bertscore_for_known_pair():
    settings = Settings(_env_file=None)
    calculator = MetricsCalculator(settings.embedding_cache_dir, settings.metric_nltk_dir)
    identical, different = calculator.calculate([
        ("The bridge reopened.", "The bridge reopened."),
        ("A father cooked dinner.", "The bridge reopened."),
    ])
    assert identical.bertscore_f1 == pytest.approx(1, abs=1e-6)
    assert different.bertscore_f1 < identical.bertscore_f1
    assert calculator.metadata()["bertscore_model"] == BERT_MODEL
