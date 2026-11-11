# -*- coding: utf-8 -*-
"""
Multi-Language Sentiment Analysis Configuration
"""

from pathlib import Path

# =============================================================================
# 📁 Project Path Settings
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Language-specific output folders
LANGUAGE_OUTPUT_DIRS = {
    'k': OUTPUT_DIR / "korea",  # Korean
    'c': OUTPUT_DIR / "china",  # Chinese
    'e': OUTPUT_DIR / "english",  # English
    'j': OUTPUT_DIR / "japan"   # Japanese
}

# =============================================================================
# 🔧 Model Settings
# =============================================================================

# Korean models
KOREAN_MODELS = {
    'primary': "snunlp/KR-FinBert-SC",                           # Main model
    'alternative': "sangrimlee/bert-base-multilingual-cased-nsmc",  # Alternative model
    'fallback': "matthewburke/korean_sentiment"                  # Fallback model
}

# Chinese models
CHINESE_MODELS = {
    'primary': "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",       # Main model
    'alternative': "uer/roberta-base-finetuned-dianping-chinese",   # Alternative model
    'fallback': "uer/roberta-base-finetuned-jd-binary-chinese"      # Fallback model
}

# English models
ENGLISH_MODELS = {
    'primary': "cardiffnlp/twitter-roberta-base-sentiment-latest",     # Main model
    'alternative': "finiteautomata/bertweet-base-sentiment-analysis",  # Alternative model
    'fallback': "distilbert-base-uncased-finetuned-sst-2-english"      # Fallback model
}

# Japanese models
JAPANESE_MODELS = {
    'primary': "christian-phu/bert-finetuned-japanese-sentiment",  # Main model
    'alternative': "koheiduck/bert-japanese-finetuned-sentiment",  # Alternative model
    'fallback': "jarvisx17/japanese-sentiment-analysis"            # Fallback model
}

# Language-to-model mapping
LANGUAGE_MODELS = {
    'k': KOREAN_MODELS,
    'c': CHINESE_MODELS,
    'e': ENGLISH_MODELS,
    'j': JAPANESE_MODELS
}

# =============================================================================
# ⚙️ Processing Settings
# =============================================================================

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    'batch_size': 16,           # Batch size
    'max_length': 128,          # Maximum text length
    'confidence_threshold': 0.8, # Confidence threshold
    'device': 'auto'            # Device auto-selection
}

# Language detection settings
LANGUAGE_DETECTION_CONFIG = {
    'fallback_language': 'e'    # Used when a sheet's language can't be detected at all
}

# =============================================================================
# 📊 Visualization Settings
# =============================================================================

# WordCloud settings
WORDCLOUD_CONFIG = {
    'width': 1600,
    'height': 800,
    'background_color': 'white',
    'max_words': 300,
    'min_font_size': 12,
    'relative_scaling': 0.5,
    'colormap': 'plasma',
    'collocations': False
}

# Chart settings
CHART_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colors': {
        'positive': ['lightgreen', 'green'],
        'neutral': ['lightgray', 'gray'],
        'negative': ['lightcoral', 'red']
    }
}

# =============================================================================
# 📝 File Settings
# =============================================================================

# Input file settings
INPUT_CONFIG = {
    'supported_formats': ['.xlsx', '.xls'],
    'text_columns': ['comment', '댓글', 'review', 'text', 'comments', '텍스트'],
    'encoding': 'utf-8'
}

# Output file settings
OUTPUT_CONFIG = {
    'format': 'xlsx',
    'include_original': True,
    'include_confidence': True,
    'include_language': True,
    'include_metadata': True
}

# =============================================================================
# 🔍 Logging Settings
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': OUTPUT_DIR / 'logs' / 'sentiment_analysis.log'
}

# =============================================================================
# 🚀 Performance Settings
# =============================================================================

# Hugging Face model cache. Passed as `cache_dir=` to from_pretrained() so
# downloaded models stay inside the project instead of ~/.cache/huggingface.
PERFORMANCE_CONFIG = {
    'cache_dir': OUTPUT_DIR / 'cache'
}

# =============================================================================
# 📋 Utility Functions
# =============================================================================

def ensure_directories():
    """Create necessary directories."""
    for lang_dir in LANGUAGE_OUTPUT_DIRS.values():
        lang_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log and cache directories under output folder
    (OUTPUT_DIR / 'logs').mkdir(exist_ok=True)
    (OUTPUT_DIR / 'cache').mkdir(exist_ok=True)

def get_model_name(language, model_type='primary'):
    """Get model name for language and model type."""
    if language in LANGUAGE_MODELS:
        return LANGUAGE_MODELS[language].get(model_type, LANGUAGE_MODELS[language]['primary'])
    return None

def get_output_path(language, filename):
    """Get output path for language and filename."""
    if language in LANGUAGE_OUTPUT_DIRS:
        return LANGUAGE_OUTPUT_DIRS[language] / filename
    return OUTPUT_DIR / filename

# Initialize directories
ensure_directories()
