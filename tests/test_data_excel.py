# -*- coding: utf-8 -*-
"""DataProcessor's pure logic: text column detection, sheet-name language
inference, and output column filtering — no Excel I/O or model calls."""

import pandas as pd
import pytest

from data_excel import DataProcessor


@pytest.fixture
def processor():
    return DataProcessor()


def test_find_text_column_by_known_name(processor):
    df = pd.DataFrame({'id': [1, 2], 'comment': ['great movie', 'not so great']})
    assert processor.find_text_column(df) == 'comment'


def test_find_text_column_falls_back_to_longest_string_column(processor):
    df = pd.DataFrame({
        'code': ['A1', 'B2'],
        'review_text': ['This was a genuinely wonderful film experience', 'Terrible pacing throughout'],
    })
    assert processor.find_text_column(df) == 'review_text'


def test_find_text_column_returns_none_when_nothing_matches(processor):
    df = pd.DataFrame({'id': [1, 2], 'score': [5, 3]})
    assert processor.find_text_column(df) is None


@pytest.mark.parametrize("sheet_name,expected", [
    ("k_sample", "k"), ("Korean Reviews", "k"), ("한국어", "k"),
    ("c_sample", "c"), ("Chinese Reviews", "c"), ("중국어", "c"),
    ("j_sample", "j"), ("Japanese Reviews", "j"), ("일본어", "j"),
    ("e_sample", "e"), ("English Reviews", "e"), ("영어", "e"),
])
def test_infer_language_from_sheet_name(processor, sheet_name, expected):
    assert processor._infer_language_from_sheet_name(sheet_name) == expected


def test_infer_language_from_sheet_name_unrecognized_uses_configured_fallback(processor):
    from core_config import LANGUAGE_DETECTION_CONFIG
    assert processor._infer_language_from_sheet_name("Sheet1") == LANGUAGE_DETECTION_CONFIG['fallback_language']


def test_apply_output_columns_keeps_everything_by_default(processor):
    df = pd.DataFrame({
        'comment': ['hi'], 'detected_language': ['e'], 'language_confidence': [0.9],
        'primary_language': ['e'], 'sentiment_label': ['positive'],
        'sentiment_confidence': [0.95], 'sentiment_model': ['some/model'],
    })
    assert list(processor._apply_output_columns(df).columns) == list(df.columns)


def test_apply_output_columns_can_drop_original_confidence_and_language(processor):
    from core_config import OUTPUT_CONFIG
    df = pd.DataFrame({
        'comment': ['hi'], 'detected_language': ['e'], 'language_confidence': [0.9],
        'primary_language': ['e'], 'sentiment_label': ['positive'],
        'sentiment_confidence': [0.95], 'sentiment_model': ['some/model'],
    })
    original = dict(OUTPUT_CONFIG)
    try:
        OUTPUT_CONFIG.update(include_original=False, include_confidence=False, include_language=False)
        result = processor._apply_output_columns(df)
        assert list(result.columns) == ['sentiment_label', 'sentiment_model']
    finally:
        OUTPUT_CONFIG.update(original)
