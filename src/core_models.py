# -*- coding: utf-8 -*-
"""
Language-specific Sentiment Analysis Model Manager
"""

import logging
import torch
from typing import Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

from core_config import LANGUAGE_MODELS, SENTIMENT_CONFIG, PERFORMANCE_CONFIG

# Logging is configured once in main.py; modules only get a named logger.
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages sentiment analysis models for different languages"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = self._get_device()
        self.model_cache = {}
        
        logger.info(f"ModelManager initialized. Device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Returns available device."""
        if SENTIMENT_CONFIG['device'] == 'auto':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("CUDA GPU available - running in GPU mode")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Apple Silicon MPS available - running in MPS mode")
            else:
                device = torch.device("cpu")
                logger.info("No GPU available - running in CPU mode")
        else:
            device = torch.device(SENTIMENT_CONFIG['device'])
        
        return device
    
    def load_model(self, language: str, model_type: str = 'primary') -> bool:
        """
        Loads model for specific language.
        
        Args:
            language (str): Language code ('k', 'c', 'e', 'j')
            model_type (str): Model type ('primary', 'alternative', 'fallback')
            
        Returns:
            bool: Load success status
        """
        try:
            if language not in LANGUAGE_MODELS:
                logger.error(f"Unsupported language: {language}")
                return False
            
            model_name = LANGUAGE_MODELS[language].get(model_type)
            if not model_name:
                logger.error(f"Model type '{model_type}' not found: {language}")
                return False
            
            # Check if model already loaded
            cache_key = f"{language}_{model_type}"
            if cache_key in self.model_cache:
                logger.info(f"Using cached model: {cache_key}")
                return True
            
            logger.info(f"Loading model: {language} - {model_type} ({model_name})")
            
            cache_dir = str(PERFORMANCE_CONFIG['cache_dir'])

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
            model.to(self.device)
            model.eval()
            
            # Save to cache
            self.model_cache[cache_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'name': model_name
            }
            
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {language} - {model_type} - {str(e)}")
            return False
    
    def get_model(self, language: str, model_type: str = 'primary') -> Optional[Dict[str, Any]]:
        """
        Returns loaded model.
        
        Args:
            language (str): Language code
            model_type (str): Model type
            
        Returns:
            Optional[Dict]: Model info or None
        """
        cache_key = f"{language}_{model_type}"
        return self.model_cache.get(cache_key)
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Loads primary models for all languages.
        
        Returns:
            Dict[str, bool]: Language-to-load-status mapping
        """
        results = {}
        
        for language in LANGUAGE_MODELS.keys():
            success = self.load_model(language, 'primary')
            results[language] = success
            
            if success:
                logger.info(f"✅ {language} model loaded successfully")
            else:
                logger.warning(f"⚠️ {language} model loading failed")
        
        return results
    
    def predict_sentiment(self, texts: list, language: str, 
                         model_type: str = 'primary') -> list:
        """
        Performs sentiment analysis on text list.
        
        Args:
            texts (list): List of texts to analyze
            language (str): Language code
            model_type (str): Model type
            
        Returns:
            list: Sentiment analysis results
        """
        # Load model if not loaded
        if not self.get_model(language, model_type):
            if not self.load_model(language, model_type):
                # Try fallback model
                if model_type != 'fallback':
                    logger.warning(f"Primary model loading failed, trying fallback: {language}")
                    if not self.load_model(language, 'fallback'):
                        logger.error(f"All models failed to load: {language}")
                        return self._get_default_results(texts)
                else:
                    logger.error(f"Fallback model also failed: {language}")
                    return self._get_default_results(texts)
        
        model_info = self.get_model(language, model_type)
        if not model_info:
            return self._get_default_results(texts)
        
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        try:
            results = []
            batch_size = SENTIMENT_CONFIG['batch_size']
            max_length = SENTIMENT_CONFIG['max_length']
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenization
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                ).to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    confidences, predictions = torch.max(probs, dim=-1)
                
                # Process results
                for pred, conf in zip(predictions.tolist(), confidences.tolist()):
                    label, confidence = self._process_prediction(
                        pred, conf, language, model_info['name']
                    )
                    results.append({
                        'label': label,
                        'confidence': round(confidence, 3),
                        'language': language,
                        'model': model_info['name']
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {language} - {str(e)}")
            return self._get_default_results(texts)
    
    def _process_prediction(self, prediction: int, confidence: float, 
                           language: str, model_name: str) -> Tuple[str, float]:
        """
        Processes model prediction results.
        
        Args:
            prediction (int): Model prediction
            confidence (float): Confidence score
            language (str): Language code
            model_name (str): Model name
            
        Returns:
            Tuple[str, float]: (Label, confidence)
        """
        # Check confidence threshold
        threshold = SENTIMENT_CONFIG['confidence_threshold']
        if confidence < threshold:
            return 'neutral', confidence
        
        # Language-specific label mapping
        label_mapping = self._get_label_mapping(language, model_name)
        
        if prediction < len(label_mapping):
            label = label_mapping[prediction]
        else:
            label = 'neutral'
        
        return label, confidence
    
    def _get_label_mapping(self, language: str, model_name: str) -> list:
        """Returns label mapping for language and model."""
        
        # Default mapping (most models)
        default_mapping = ['negative', 'neutral', 'positive']
        
        # Model-specific mapping
        model_specific_mapping = {
            # Korean models
            'snunlp/KR-FinBert-SC': ['negative', 'positive', 'neutral'],  # 3-class
            'sangrimlee/bert-base-multilingual-cased-nsmc': ['negative', 'positive'],  # 2-class
            'matthewburke/korean_sentiment': ['negative', 'positive'],  # 2-class

            # Chinese models (all 2-class)
            'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment': ['negative', 'positive'],
            'uer/roberta-base-finetuned-dianping-chinese': ['negative', 'positive'],
            'uer/roberta-base-finetuned-jd-binary-chinese': ['negative', 'positive'],

            # English model (2-class)
            'distilbert-base-uncased-finetuned-sst-2-english': ['negative', 'positive'],

            # Japanese models
            'koheiduck/bert-japanese-finetuned-sentiment': ['neutral', 'negative', 'positive'],  # 3-class, non-default order
            'jarvisx17/japanese-sentiment-analysis': ['negative', 'positive'],  # 2-class
        }
        
        return model_specific_mapping.get(model_name, default_mapping)
    
    def _get_default_results(self, texts: list) -> list:
        """Returns default results (when model loading fails)."""
        return [{
            'label': 'neutral',
            'confidence': 0.0,
            'language': 'unknown',
            'model': 'none'
        } for _ in texts]
    
    def get_loaded_models(self) -> list:
        """Returns list of loaded models."""
        return list(self.model_cache.keys())
    
    def unload_model(self, language: str, model_type: str = 'primary'):
        """Unloads specific model."""
        cache_key = f"{language}_{model_type}"
        if cache_key in self.model_cache:
            del self.model_cache[cache_key]
            logger.info(f"Model unloaded: {cache_key}")
    
    def clear_cache(self):
        """Clears all model cache."""
        self.model_cache.clear()
        logger.info("All model cache cleared")
    
    def get_model_info(self, language: str, model_type: str = 'primary') -> Optional[Dict]:
        """Returns model information."""
        model_info = self.get_model(language, model_type)
        if model_info:
            return {
                'name': model_info['name'],
                'device': str(self.device),
                'loaded': True
            }
        return None

# Global instance
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """Returns ModelManager instance."""
    return model_manager
