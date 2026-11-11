# -*- coding: utf-8 -*-
"""
Language Auto-Detection Module
"""

import re
import logging
from typing import Tuple
from langdetect import detect, DetectorFactory, LangDetectException

# Logging is configured once in main.py; modules only get a named logger.
logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0

class LanguageDetector:
    """Auto-detects language of multilingual text"""
    
    def __init__(self):
        # Language-specific character patterns
        self.language_patterns = {
            'k': {  # Korean
                'patterns': [
                    r'[가-힣]',  # Hangul
                    r'[ㄱ-ㅎㅏ-ㅣ]',  # Hangul jamo
                ],
                'keywords': ['이', '가', '을', '를', '은', '는', '의', '에', '에서', '로', '으로'],
                'confidence': 0.9
            },
            'c': {  # Chinese
                'patterns': [
                    r'[\u4e00-\u9fff]',  # Han characters
                    r'[\u3400-\u4dbf]',  # Han extension A
                ],
                'keywords': ['的', '了', '在', '是', '有', '和', '与', '或', '但', '而'],
                'confidence': 0.9
            },
            'j': {  # Japanese
                'patterns': [
                    r'[\u3040-\u309f]',  # Hiragana
                    r'[\u30a0-\u30ff]',  # Katakana
                    r'[\u4e00-\u9fff]',  # Han characters
                ],
                'keywords': ['は', 'が', 'を', 'に', 'へ', 'で', 'と', 'から', 'まで', 'より'],
                'confidence': 0.9
            },
            'e': {  # English
                'patterns': [
                    r'[a-zA-Z]',  # English letters
                ],
                'keywords': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'],
                'confidence': 0.8
            }
        }
        
        # Language code mapping
        self.lang_code_mapping = {
            'ko': 'k', 'korean': 'k',
            'zh': 'c', 'chinese': 'c', 'zh-cn': 'c', 'zh-tw': 'c',
            'ja': 'j', 'japanese': 'j',
            'en': 'e', 'english': 'e'
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detects language of text.
        
        Args:
            text (str): Text to detect
            
        Returns:
            Tuple[str, float]: (Language code, confidence)
        """
        if not text or not text.strip():
            return 'e', 0.0  # Default: English
        
        text = text.strip()
        
        # 1. Pattern-based detection
        pattern_result = self._detect_by_patterns(text)
        
        # 2. Use langdetect library
        try:
            langdetect_result = detect(text)
            langdetect_code = self.lang_code_mapping.get(langdetect_result, langdetect_result)
            langdetect_confidence = 0.7
        except LangDetectException:
            langdetect_code = 'e'
            langdetect_confidence = 0.0
        
        # 3. Combine results
        final_language, final_confidence = self._combine_results(
            pattern_result, (langdetect_code, langdetect_confidence)
        )
        
        logger.info(f"Language detection result: {text[:50]}... -> {final_language} (confidence: {final_confidence:.2f})")
        
        return final_language, final_confidence
    
    def _detect_by_patterns(self, text: str) -> Tuple[str, float]:
        """Pattern-based language detection"""
        scores = {}
        
        for lang_code, lang_info in self.language_patterns.items():
            score = 0.0
            
            # Pattern matching score
            for pattern in lang_info['patterns']:
                matches = len(re.findall(pattern, text))
                if matches > 0:
                    score += (matches / len(text)) * 0.6
            
            # Keyword matching score
            for keyword in lang_info['keywords']:
                if keyword in text:
                    score += 0.4
            
            scores[lang_code] = min(score, 1.0)
        
        # Select language with highest score
        if scores:
            best_lang = max(scores, key=scores.get)
            best_score = scores[best_lang]
            return best_lang, best_score
        
        return 'e', 0.0
    
    def _combine_results(self, pattern_result: Tuple[str, float], 
                        langdetect_result: Tuple[str, float]) -> Tuple[str, float]:
        """Combines pattern-based and langdetect results"""
        pattern_lang, pattern_conf = pattern_result
        detect_lang, detect_conf = langdetect_result
        
        # Use pattern-based result if high confidence
        if pattern_conf > 0.8:
            return pattern_lang, pattern_conf
        
        # Use langdetect result if high confidence
        if detect_conf > 0.7:
            return detect_lang, detect_conf
        
        # If both results match
        if pattern_lang == detect_lang:
            combined_conf = (pattern_conf + detect_conf) / 2
            return pattern_lang, combined_conf
        
        # Prefer pattern-based result (more accurate)
        return pattern_lang, pattern_conf
    
    def batch_detect(self, texts: list) -> list:
        """
        Batch detects languages for multiple texts.
        
        Args:
            texts (list): List of texts to detect
            
        Returns:
            list: List of (language code, confidence) tuples
        """
        results = []
        for text in texts:
            result = self.detect_language(text)
            results.append(result)
        return results
    
    def get_language_name(self, lang_code: str) -> str:
        """Converts language code to language name"""
        language_names = {
            'k': 'Korean',
            'c': 'Chinese',
            'j': 'Japanese',
            'e': 'English'
        }
        return language_names.get(lang_code, 'Unknown')
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Checks if language is supported"""
        return lang_code in ['k', 'c', 'j', 'e']

# Global instance
language_detector = LanguageDetector()

def detect_language(text: str) -> Tuple[str, float]:
    """Language detection function (convenience)"""
    return language_detector.detect_language(text)

def batch_detect_languages(texts: list) -> list:
    """Batch language detection function (convenience)"""
    return language_detector.batch_detect(texts)

def get_language_name(lang_code: str) -> str:
    """Converts language code to language name (module-level function)"""
    return language_detector.get_language_name(lang_code)
