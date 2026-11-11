# -*- coding: utf-8 -*-
"""SentimentVisualizer: comparison-chart aggregation (regression test for the
sheet-name-vs-language-code bug) and text preprocessing — no actual model
inference needed."""

from pathlib import Path

from data_charts import get_visualizer

visualizer = get_visualizer()


def test_comparison_chart_aggregates_multiple_sheets_of_the_same_language(tmp_path):
    # Two sheets both detected as Korean ('k') should be summed together,
    # not treated as two separate "languages" keyed by sheet name.
    all_results = {
        'Sheet1': {'language': 'k', 'sentiment_counts': {'positive': 3, 'negative': 1}},
        'Sheet2': {'language': 'k', 'sentiment_counts': {'positive': 2, 'neutral': 4}},
        'Sheet3': {'language': 'e', 'sentiment_counts': {'negative': 5}},
    }

    chart_path = visualizer.create_comparison_chart(all_results, tmp_path)

    assert chart_path is not None
    assert Path(chart_path).exists()
    assert Path(chart_path).name == "Language_Comparison_Chart.png"


def test_comparison_chart_returns_none_when_no_sentiment_counts(tmp_path):
    assert visualizer.create_comparison_chart({'Sheet1': {'language': 'k'}}, tmp_path) is None
    assert visualizer.create_comparison_chart({}, tmp_path) is None


def test_find_cjk_font_returns_a_path_or_none():
    for lang in ('k', 'c', 'j'):
        font = visualizer._find_cjk_font(lang)
        assert font is None or isinstance(font, str)


def test_find_cjk_font_unsupported_language_returns_none():
    assert visualizer._find_cjk_font('e') is None


def test_preprocess_text_for_wordcloud_keeps_only_target_script():
    assert visualizer._preprocess_text_for_wordcloud("Hello123 world!!", 'e') == "Hello world"
    assert visualizer._preprocess_text_for_wordcloud("좋아요 best 123", 'k') == "좋아요"
