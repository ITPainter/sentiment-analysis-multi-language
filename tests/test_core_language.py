# -*- coding: utf-8 -*-
"""Language detection: pattern-based + langdetect-combined results."""

from core_language import LanguageDetector

detector = LanguageDetector()


def test_detects_korean():
    lang, confidence = detector.detect_language("이 영화 정말 재미있었어요! 배우들의 연기가 훌륭했습니다.")
    assert lang == 'k'
    assert confidence > 0


def test_detects_chinese():
    lang, confidence = detector.detect_language("这部电影很有趣！演员的表演很出色，特效令人印象深刻。")
    assert lang == 'c'
    assert confidence > 0


def test_detects_japanese():
    lang, confidence = detector.detect_language("この映画は本当に面白かったです。ストーリーがとても良かったです。")
    assert lang == 'j'
    assert confidence > 0


def test_detects_english():
    lang, confidence = detector.detect_language("This movie was really entertaining and the acting was excellent.")
    assert lang == 'e'
    assert confidence > 0


def test_empty_text_defaults_to_english_with_zero_confidence():
    assert detector.detect_language("") == ('e', 0.0)
    assert detector.detect_language("   ") == ('e', 0.0)


def test_batch_detect_matches_single_detect():
    texts = ["안녕하세요", "Hello there"]
    assert detector.batch_detect(texts) == [detector.detect_language(t) for t in texts]


def test_get_language_name():
    assert detector.get_language_name('k') == 'Korean'
    assert detector.get_language_name('c') == 'Chinese'
    assert detector.get_language_name('j') == 'Japanese'
    assert detector.get_language_name('e') == 'English'
    assert detector.get_language_name('xx') == 'Unknown'


def test_is_supported_language():
    assert detector.is_supported_language('k')
    assert not detector.is_supported_language('xx')
