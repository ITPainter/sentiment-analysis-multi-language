# -*- coding: utf-8 -*-
"""Config sanity checks: every language is wired consistently and
get_model_name()/get_output_path() behave as documented."""

import core_config as cfg


def test_all_languages_have_output_dirs_and_models():
    assert set(cfg.LANGUAGE_OUTPUT_DIRS) == set(cfg.LANGUAGE_MODELS) == {'k', 'c', 'e', 'j'}


def test_every_model_entry_is_a_non_empty_string():
    for lang, models in cfg.LANGUAGE_MODELS.items():
        for model_type in ('primary', 'alternative', 'fallback'):
            assert model_type in models, f"{lang} is missing a '{model_type}' model"
            assert isinstance(models[model_type], str) and models[model_type], \
                f"{lang}.{model_type} must be a non-empty model id"


def test_get_model_name_returns_primary_by_default():
    for lang in cfg.LANGUAGE_MODELS:
        assert cfg.get_model_name(lang) == cfg.LANGUAGE_MODELS[lang]['primary']


def test_get_model_name_unknown_language_returns_none():
    assert cfg.get_model_name('xx') is None


def test_get_output_path_uses_language_dir():
    path = cfg.get_output_path('k', 'result.xlsx')
    assert path == cfg.LANGUAGE_OUTPUT_DIRS['k'] / 'result.xlsx'


def test_get_output_path_unknown_language_falls_back_to_output_dir():
    path = cfg.get_output_path('xx', 'result.xlsx')
    assert path == cfg.OUTPUT_DIR / 'result.xlsx'
