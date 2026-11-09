# -*- coding: utf-8 -*-
"""
Visualization Module
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

from core_config import WORDCLOUD_CONFIG, CHART_CONFIG, LANGUAGE_OUTPUT_DIRS

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# matplotlib settings
import matplotlib.font_manager as fm

# 폰트 설정 (macOS 환경에 최적화)
plt.rcParams['font.family'] = ['Apple SD Gothic Neo', 'AppleGothic', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 폰트 경고 메시지 차단
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 한글 폰트 강제 설정
def setup_korean_font():
    """한글 폰트 강제 설정"""
    try:
        # Apple SD Gothic Neo 폰트 찾기
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        if os.path.exists(font_path):
            # matplotlib에 폰트 등록
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = ['Apple SD Gothic Neo', 'AppleGothic']
            logger.info(f"한글 폰트 설정 완료: {font_path}")
            return True
        else:
            logger.warning("Apple SD Gothic Neo 폰트를 찾을 수 없습니다")
            return False
    except Exception as e:
        logger.error(f"한글 폰트 설정 실패: {str(e)}")
        return False

# 한글 폰트 설정 실행
setup_korean_font()

class SentimentVisualizer:
    """Creates visualizations for sentiment analysis results"""
    
    def __init__(self):
        self.wordcloud_config = WORDCLOUD_CONFIG
        self.chart_config = CHART_CONFIG
        
        # Language-specific colors
        self.language_colors = {
            'k': {'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'},
            'c': {'positive': '#FF9800', 'neutral': '#9E9E9E', 'negative': '#E91E63'},
            'j': {'positive': '#2196F3', 'neutral': '#9E9E9E', 'negative': '#9C27B0'},
            'e': {'positive': '#8BC34A', 'neutral': '#9E9E9E', 'negative': '#FF5722'}
        }
        
        logger.info("SentimentVisualizer initialized")
    
    def create_sentiment_charts(self, df: pd.DataFrame, language: str, 
                               output_dir: Path, filename_prefix: str) -> Dict[str, str]:
        """
        Creates charts for sentiment analysis results.
        
        Args:
            df (pd.DataFrame): Sentiment analysis results DataFrame
            language (str): Language code
            output_dir (Path): Output directory
            filename_prefix (str): Filename prefix
            
        Returns:
            Dict[str, str]: Generated chart file paths
        """
        try:
            # Check sentiment analysis results column
            if 'sentiment_label' not in df.columns:
                logger.warning("Sentiment analysis results column 'sentiment_label' not found")
                return {}
            
            # Results statistics
            sentiment_counts = df['sentiment_label'].value_counts()
            
            # Chart file paths
            chart_files = {}
            
            # 1. Bar chart
            bar_chart_path = self._create_bar_chart(
                sentiment_counts, language, output_dir, f"{filename_prefix}_bar_chart"
            )
            if bar_chart_path:
                chart_files['bar_chart'] = bar_chart_path
            
            # 2. Pie chart
            pie_chart_path = self._create_pie_chart(
                sentiment_counts, language, output_dir, f"{filename_prefix}_pie_chart"
            )
            if pie_chart_path:
                chart_files['pie_chart'] = pie_chart_path
            
            # 3. WordCloud (if text column exists)
            text_column = self._find_text_column(df)
            if text_column:
                wordcloud_path = self._create_wordcloud(
                    df, text_column, language, output_dir, f"{filename_prefix}_wordcloud"
                )
                if wordcloud_path:
                    chart_files['wordcloud'] = wordcloud_path
            
            # 4. Confidence distribution histogram
            if 'sentiment_confidence' in df.columns:
                confidence_path = self._create_confidence_histogram(
                    df, language, output_dir, f"{filename_prefix}_confidence"
                )
                if confidence_path:
                    chart_files['confidence'] = confidence_path
            
            logger.info(f"Chart creation completed: {len(chart_files)} charts")
            return chart_files
            
        except Exception as e:
            logger.error(f"Chart creation failed: {str(e)}")
            return {}
    
    def _create_bar_chart(self, sentiment_counts: pd.Series, language: str, 
                          output_dir: Path, filename: str) -> Optional[str]:
        """Creates bar chart"""
        try:
            plt.figure(figsize=self.chart_config['figure_size'])
            
            # Color settings
            colors = [self.language_colors[language].get(label, '#9E9E9E') 
                     for label in sentiment_counts.index]
            
            # Create bar chart
            bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
            
            # Title and labels
            lang_name = {'k': 'Korean', 'c': 'Chinese', 'j': 'Japanese', 'e': 'English'}.get(language, language)
            plt.title(f'{lang_name} Sentiment Analysis Results', fontsize=16, fontweight='bold', fontfamily='Apple SD Gothic Neo')
            plt.xlabel('Sentiment', fontsize=12, fontfamily='Apple SD Gothic Neo')
            plt.ylabel('Text Count', fontsize=12, fontfamily='Apple SD Gothic Neo')
            
            # Display values on bars
            for bar, count in zip(bars, sentiment_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save file
            output_path = output_dir / f"{filename}.png"
            plt.savefig(output_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Bar chart created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Bar chart creation failed: {str(e)}")
            return None
    
    def _create_pie_chart(self, sentiment_counts: pd.Series, language: str, 
                          output_dir: Path, filename: str) -> Optional[str]:
        """Creates pie chart"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Color settings
            colors = [self.language_colors[language].get(label, '#9E9E9E') 
                     for label in sentiment_counts.index]
            
            # Create pie chart
            wedges, texts, autotexts = plt.pie(
                sentiment_counts.values, 
                labels=sentiment_counts.index,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Title
            lang_name = {'k': 'Korean', 'c': 'Chinese', 'j': 'Japanese', 'e': 'English'}.get(language, language)
            plt.title(f'{lang_name} Sentiment Analysis Ratio', fontsize=16, fontweight='bold', fontfamily='Apple SD Gothic Neo')
            
            # Text styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.axis('equal')
            plt.tight_layout()
            
            # Save file
            output_path = output_dir / f"{filename}.png"
            plt.savefig(output_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pie chart created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Pie chart creation failed: {str(e)}")
            return None
    
    def _create_wordcloud(self, df: pd.DataFrame, text_column: str, language: str, 
                          output_dir: Path, filename: str) -> Optional[str]:
        """Creates wordcloud"""
        try:
            # Extract and preprocess text data
            texts = df[text_column].dropna().astype(str).tolist()
            if not texts:
                return None
            
            # Combine text
            combined_text = ' '.join(texts)
            
            # Language-specific preprocessing
            processed_text = self._preprocess_text_for_wordcloud(combined_text, language)
            
            if not processed_text.strip():
                return None
            
            # 폰트 설정 (macOS 환경에 최적화)
            font_path = None
            if language == 'k':  # Korean
                # macOS 기본 한국어 폰트들 (우선순위 순)
                korean_fonts = [
                    '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # 가장 안정적인 한글 폰트
                    '/System/Library/Fonts/AppleGothic.ttf',
                    '/System/Library/AssetsV2/com_apple_MobileAsset_Font7/de4b2bad515a67ab2d11e39fd896b1e189252a43.asset/AssetData/NanumGothic.ttc'  # 나눔고딕
                ]
                for font in korean_fonts:
                    if os.path.exists(font):
                        font_path = font
                        logger.info(f"한글 폰트 사용: {font}")
                        break
                if not font_path:
                    logger.warning("한글 폰트를 찾을 수 없어 기본 폰트 사용")
            elif language == 'c':  # Chinese
                # macOS 기본 중국어 폰트들
                chinese_fonts = [
                    '/System/Library/Fonts/PingFang.ttc',
                    '/System/Library/Fonts/STHeiti Light.ttc',
                    '/System/Library/Fonts/STHeiti Medium.ttc'
                ]
                for font in chinese_fonts:
                    if os.path.exists(font):
                        font_path = font
                        break
            elif language == 'j':  # Japanese
                # macOS 기본 일본어 폰트들
                japanese_fonts = [
                    '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
                    '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
                    '/System/Library/Fonts/ヒラギノ明朝 W3.ttc'
                ]
                for font in japanese_fonts:
                    if os.path.exists(font):
                        font_path = font
                        break
            
            # 폰트 파일 존재 확인
            if font_path and os.path.exists(font_path):
                logger.info(f"Using font: {font_path}")
            else:
                font_path = None
                logger.info("Using default font")
            
            # Create wordcloud
            wordcloud = WordCloud(
                width=self.wordcloud_config['width'],
                height=self.wordcloud_config['height'],
                background_color=self.wordcloud_config['background_color'],
                max_words=self.wordcloud_config['max_words'],
                min_font_size=self.wordcloud_config['min_font_size'],
                relative_scaling=self.wordcloud_config['relative_scaling'],
                colormap=self.wordcloud_config['colormap'],
                collocations=self.wordcloud_config['collocations'],
                font_path=font_path  # 폰트 경로 설정
            ).generate(processed_text)
            
            # Visualization
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Title
            lang_name = {'k': 'Korean', 'c': 'Chinese', 'j': 'Japanese', 'e': 'English'}.get(language, language)
            plt.title(f'{lang_name} Text WordCloud', fontsize=20, fontweight='bold', fontfamily='Apple SD Gothic Neo')
            
            plt.tight_layout()
            
            # Save file
            output_path = output_dir / f"{filename}.png"
            plt.savefig(output_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"WordCloud created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"WordCloud creation failed: {str(e)}")
            return None
    
    def _create_confidence_histogram(self, df: pd.DataFrame, language: str, 
                                   output_dir: Path, filename: str) -> Optional[str]:
        """Creates confidence distribution histogram"""
        try:
            plt.figure(figsize=self.chart_config['figure_size'])
            
            # Confidence data
            confidence_data = df['sentiment_confidence'].dropna()
            
            if len(confidence_data) == 0:
                return None
            
            # Create histogram
            plt.hist(confidence_data, bins=20, alpha=0.7, color=self.language_colors[language]['neutral'])
            
            # Title and labels
            lang_name = {'k': 'Korean', 'c': 'Chinese', 'j': 'Japanese', 'e': 'English'}.get(language, language)
            plt.title(f'{lang_name} Sentiment Analysis Confidence Distribution', fontsize=16, fontweight='bold', fontfamily='Apple SD Gothic Neo')
            plt.xlabel('Confidence', fontsize=12, fontfamily='Apple SD Gothic Neo')
            plt.ylabel('Frequency', fontsize=12, fontfamily='Apple SD Gothic Neo')
            
            # Add mean line
            mean_confidence = confidence_data.mean()
            plt.axvline(mean_confidence, color='red', linestyle='--', 
                       label=f'Mean: {mean_confidence:.3f}')
            plt.legend()
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save file
            output_path = output_dir / f"{filename}.png"
            plt.savefig(output_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confidence histogram created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Confidence histogram creation failed: {str(e)}")
            return None
    
    def _preprocess_text_for_wordcloud(self, text: str, language: str) -> str:
        """Text preprocessing for wordcloud"""
        import re
        
        # Basic preprocessing
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)       # Remove consecutive spaces
        
        # Language-specific preprocessing
        if language == 'k':  # Korean
            # Keep only Hangul
            text = re.sub(r'[^가-힣\s]', ' ', text)
        elif language == 'c':  # Chinese
            # Keep only Han characters
            text = re.sub(r'[^\u4e00-\u9fff\s]', ' ', text)
        elif language == 'j':  # Japanese
            # Keep only Hiragana, Katakana, Han characters
            text = re.sub(r'[^あ-んア-ン一-龯\s]', ' ', text)
        elif language == 'e':  # English
            # Keep only English letters
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        return text.strip()
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Finds text column"""
        text_columns = ['comment', '댓글', 'review', 'text', 'comments', '텍스트']
        
        for col in df.columns:
            if col.lower() in [tc.lower() for tc in text_columns]:
                return col
        
        # Estimate by content
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_data = df[col].dropna().head(5)
                if len(sample_data) > 0:
                    avg_length = sample_data.astype(str).str.len().mean()
                    if avg_length > 10:
                        return col
        
        return None
    
    def create_comparison_chart(self, all_results: Dict[str, Dict], 
                               output_dir: Path) -> Optional[str]:
        """
        Creates comparison chart for all languages.
        
        Args:
            all_results (Dict): All language results
            output_dir (Path): Output directory
            
        Returns:
            Optional[str]: Created chart file path
        """
        try:
            # Aggregate sentiment counts by language (all_results is keyed by sheet name,
            # and multiple sheets can share the same language)
            language_totals: Dict[str, Dict[str, int]] = {}
            for sheet_data in all_results.values():
                if 'sentiment_counts' not in sheet_data or 'language' not in sheet_data:
                    continue
                lang = sheet_data['language']
                totals = language_totals.setdefault(lang, {'positive': 0, 'neutral': 0, 'negative': 0})
                for label, count in sheet_data['sentiment_counts'].items():
                    totals[label] = totals.get(label, 0) + count

            languages = list(language_totals.keys())
            positive_counts = [language_totals[lang]['positive'] for lang in languages]
            neutral_counts = [language_totals[lang]['neutral'] for lang in languages]
            negative_counts = [language_totals[lang]['negative'] for lang in languages]

            if not languages:
                return None
            
            # Create comparison chart
            x = np.arange(len(languages))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            bars1 = ax.bar(x - width, positive_counts, width, label='Positive', 
                          color='lightgreen', alpha=0.8)
            bars2 = ax.bar(x, neutral_counts, width, label='Neutral', 
                          color='lightgray', alpha=0.8)
            bars3 = ax.bar(x + width, negative_counts, width, label='Negative', 
                          color='lightcoral', alpha=0.8)
            
            # Title and labels
            ax.set_xlabel('Language', fontsize=12, fontfamily='Apple SD Gothic Neo')
            ax.set_ylabel('Text Count', fontsize=12, fontfamily='Apple SD Gothic Neo')
            ax.set_title('Language Comparison - Sentiment Analysis Results', fontsize=16, fontweight='bold', fontfamily='Apple SD Gothic Neo')
            ax.set_xticks(x)
            ax.set_xticklabels([{'k': 'Korean', 'c': 'Chinese', 'j': 'Japanese', 'e': 'English'}.get(lang, lang) 
                               for lang in languages])
            ax.legend()
            
            # Display values on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save file
            output_path = output_dir / "Language_Comparison_Chart.png"
            plt.savefig(output_path, dpi=self.chart_config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison chart created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Comparison chart creation failed: {str(e)}")
            return None

# Global instance
visualizer = SentimentVisualizer()

def get_visualizer() -> SentimentVisualizer:
    """Returns SentimentVisualizer instance."""
    return visualizer
