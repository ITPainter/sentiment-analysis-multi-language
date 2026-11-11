# -*- coding: utf-8 -*-
"""ModelManager's pure logic (label mapping, thresholding, default results) —
no actual model download/inference, so this runs offline in CI."""

from core_config import SENTIMENT_CONFIG
from core_models import ModelManager


def test_default_label_mapping_for_unknown_model():
    mm = ModelManager()
    assert mm._get_label_mapping('e', 'some/unknown-model') == ['negative', 'neutral', 'positive']


def test_known_two_class_model_mapping():
    mm = ModelManager()
    assert mm._get_label_mapping('e', 'distilbert-base-uncased-finetuned-sst-2-english') == \
        ['negative', 'positive']


def test_japanese_alternative_model_has_non_default_label_order():
    mm = ModelManager()
    mapping = mm._get_label_mapping('j', 'koheiduck/bert-japanese-finetuned-sentiment')
    assert mapping == ['neutral', 'negative', 'positive']


def test_process_prediction_below_threshold_forces_neutral():
    mm = ModelManager()
    low_confidence = SENTIMENT_CONFIG['confidence_threshold'] - 0.1
    label, confidence = mm._process_prediction(0, low_confidence, 'e', 'some/unknown-model')
    assert label == 'neutral'
    assert confidence == low_confidence


def test_process_prediction_above_threshold_uses_label_mapping():
    mm = ModelManager()
    high_confidence = min(SENTIMENT_CONFIG['confidence_threshold'] + 0.1, 0.99)
    label, confidence = mm._process_prediction(2, high_confidence, 'e', 'some/unknown-model')
    assert label == 'positive'  # index 2 in the default ['negative','neutral','positive'] mapping


def test_default_results_are_neutral_zero_confidence():
    mm = ModelManager()
    results = mm._get_default_results(["a", "b", "c"])
    assert len(results) == 3
    assert all(r == {'label': 'neutral', 'confidence': 0.0, 'language': 'unknown', 'model': 'none'}
               for r in results)
