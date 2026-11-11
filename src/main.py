# -*- coding: utf-8 -*-
"""
Multi-Language Sentiment Analysis Main Execution File
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_config import ensure_directories, LOGGING_CONFIG
from core_models import get_model_manager
from data_excel import DataProcessor
from data_charts import get_visualizer
from core_language import get_language_name

# Logging setup — the only place in the codebase that calls basicConfig,
# so every other module's logger inherits this configuration.
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level'], logging.INFO),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file'], encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultiLanguageSentimentAnalyzer:
    """Main class for multi-language sentiment analysis system"""
    
    def __init__(self):
        self.model_manager = get_model_manager()
        self.data_processor = DataProcessor()
        self.visualizer = get_visualizer()
        
        logger.info("Multi-language sentiment analysis system initialized")
    
    def run_analysis(self, input_file: str, output_dir: Optional[str] = None,
                      forced_language: Optional[str] = None) -> bool:
        """
        Runs sentiment analysis.

        Args:
            input_file (str): Input Excel file path
            output_dir (Optional[str]): Output directory (default: config output directory)
            forced_language (Optional[str]): Skip auto-detection and analyze
                every sheet as this language code (from --lang).

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"🚀 Starting sentiment analysis: {input_file}")

            # 1. Process data
            logger.info("📊 Processing Excel file...")
            processing_results = self.data_processor.process_excel_file(input_file, forced_language)

            if not processing_results['success']:
                logger.error(f"Data processing failed: {processing_results.get('error', 'Unknown error')}")
                return False

            # 2. Perform sentiment analysis (models are loaded lazily, per
            # language, the first time that language is actually needed)
            logger.info("🧠 Performing sentiment analysis...")
            analysis_results = self._perform_sentiment_analysis(processing_results)

            # 3. Save results
            logger.info("💾 Saving results...")
            saved_files = self.data_processor.save_results(processing_results, output_dir)

            # 4. Create visualizations
            logger.info("🎨 Creating visualizations...")
            visualization_results = self._create_visualizations(analysis_results, output_dir)

            # 5. Print summary
            self._print_summary(processing_results, analysis_results, saved_files, visualization_results)
            
            logger.info("🎉 Sentiment analysis completed!")
            return True
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return False
    
    def _perform_sentiment_analysis(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs sentiment analysis.
        
        Args:
            processing_results (Dict): Data processing results
            
        Returns:
            Dict: Sentiment analysis results
        """
        analysis_results = {}
        
        for sheet_name, sheet_data in processing_results['sheets'].items():
            try:
                language = sheet_data['language']
                df = sheet_data['data']
                
                # Find text column
                text_column = self.data_processor.find_text_column(df)
                if text_column is None:
                    logger.warning(f"Text column not found in sheet '{sheet_name}'")
                    continue
                
                # Extract text data
                texts = df[text_column].dropna().astype(str).tolist()
                if not texts:
                    logger.warning(f"No text data in sheet '{sheet_name}'")
                    continue
                
                logger.info(f"Analyzing sheet '{sheet_name}': {len(texts)} texts")
                
                # Perform sentiment analysis
                sentiment_results = self.model_manager.predict_sentiment(texts, language)
                
                # Add results to DataFrame
                result_df = df.copy()
                result_df['sentiment_label'] = [r['label'] for r in sentiment_results]
                result_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
                result_df['sentiment_model'] = [r['model'] for r in sentiment_results]
                
                # Save analysis results
                analysis_results[sheet_name] = {
                    'data': result_df,
                    'language': language,
                    'sentiment_results': sentiment_results,
                    'sentiment_counts': result_df['sentiment_label'].value_counts().to_dict()
                }
                
                # Update original data
                processing_results['sheets'][sheet_name]['data'] = result_df
                
                logger.info(f"Sheet '{sheet_name}' sentiment analysis completed")
                
            except Exception as e:
                logger.error(f"Sheet '{sheet_name}' sentiment analysis failed: {str(e)}")
                continue
        
        return analysis_results
    
    def _create_visualizations(self, analysis_results: Dict[str, Any], 
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates visualizations.
        
        Args:
            analysis_results (Dict): Sentiment analysis results
            output_dir (Optional[str]): Output directory
            
        Returns:
            Dict: Visualization results
        """
        visualization_results = {}
        
        for sheet_name, sheet_data in analysis_results.items():
            try:
                language = sheet_data['language']
                df = sheet_data['data']
                
                # Determine language-specific output directory
                if output_dir:
                    lang_output_dir = Path(output_dir) / language
                else:
                    from core_config import LANGUAGE_OUTPUT_DIRS
                    lang_output_dir = LANGUAGE_OUTPUT_DIRS.get(language, Path('output') / language)
                
                # Create directory
                lang_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Filename prefix
                filename_prefix = f"{sheet_name}_sentiment"
                
                # Create charts
                chart_files = self.visualizer.create_sentiment_charts(
                    df, language, lang_output_dir, filename_prefix
                )
                
                visualization_results[sheet_name] = {
                    'language': language,
                    'charts': chart_files
                }
                
                logger.info(f"Sheet '{sheet_name}' visualization completed: {len(chart_files)} charts")
                
            except Exception as e:
                logger.error(f"Sheet '{sheet_name}' visualization failed: {str(e)}")
                continue
        
        # Create overall comparison chart
        if len(analysis_results) > 1:
            try:
                if output_dir:
                    comparison_output_dir = Path(output_dir)
                else:
                    comparison_output_dir = project_root / 'output'
                
                comparison_chart = self.visualizer.create_comparison_chart(
                    analysis_results, comparison_output_dir
                )
                
                if comparison_chart:
                    visualization_results['comparison'] = {
                        'chart': comparison_chart
                    }
                    logger.info("Overall comparison chart created")
                    
            except Exception as e:
                logger.error(f"Overall comparison chart creation failed: {str(e)}")
        
        return visualization_results
    
    def _print_summary(self, processing_results: Dict[str, Any], 
                       analysis_results: Dict[str, Any], 
                       saved_files: Dict[str, str], 
                       visualization_results: Dict[str, Any]):
        """Prints results summary."""
        
        print("\n" + "="*80)
        print("🎯 Multi-Language Sentiment Analysis Results Summary")
        print("="*80)
        
        # Data processing summary
        print("\n📊 Data Processing Summary:")
        print(processing_results['summary'])
        
        # Sentiment analysis summary
        print("\n🧠 Sentiment Analysis Summary:")
        for sheet_name, sheet_data in analysis_results.items():
            language = sheet_data['language']
            lang_name = get_language_name(language)
            sentiment_counts = sheet_data['sentiment_counts']
            
            print(f"\n  📋 {sheet_name} ({lang_name}):")
            total = sum(sentiment_counts.values())
            for label, count in sentiment_counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"    - {label}: {count} texts ({percentage:.1f}%)")
        
        # Saved files
        print("\n💾 Saved Result Files:")
        for sheet_name, file_path in saved_files.items():
            print(f"  - {sheet_name}: {file_path}")
        
        # Created charts
        print("\n🎨 Created Visualizations:")
        for sheet_name, viz_data in visualization_results.items():
            if sheet_name != 'comparison':
                language = viz_data['language']
                lang_name = get_language_name(language)
                chart_count = len(viz_data['charts'])
                print(f"  - {sheet_name} ({lang_name}): {chart_count} charts")
        
        if 'comparison' in visualization_results:
            print(f"  - Overall Comparison: 1 chart")
        
        print("\n" + "="*80)

def show_help():
    """Show usage help"""
    print("🎯 Multi-Language Sentiment Analysis System")
    print("=" * 50)
    print()
    print("📋 Usage:")
    print("  python main.py <excel_file>        # Analyze Excel file from output/ folder")
    print("  python main.py                     # Create sample data and run analysis")
    print("  python main.py --sample            # Force create sample data")
    print("  python main.py --help              # Show this help")
    print()
    print("📁 Examples:")
    print("  python main.py my_data.xlsx        # Analyze my_data.xlsx from output/ folder")
    print("  python main.py                     # Create and analyze sample data")
    print("  python main.py --sample            # Force create sample data")
    print()
    print("💡 Tips:")
    print("  - Place your Excel files in the output/ folder")
    print("  - Make sure your Excel file has a text column (comment, review, etc.)")
    print("  - The system will automatically detect languages")
    print("  - Results will be saved in output/ folder by language")
    print()
    print("🔧 Advanced Options:")
    print("  python main.py file.xlsx -o /path  # Custom output directory")
    print("  python main.py file.xlsx --lang k  # Force language (k/c/e/j) instead of auto-detect")
    print("  python main.py file.xlsx --debug   # Enable debug mode")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Language Sentiment Analysis System')
    parser.add_argument('input_file', nargs='?', help='Input Excel file path (optional: will create sample data if not provided)')
    parser.add_argument('-o', '--output', help='Output directory (default: config output directory)')
    parser.add_argument('--lang', choices=['k', 'c', 'e', 'j'],
                         help='Skip language auto-detection and analyze every sheet as this language '
                              '(k=Korean, c=Chinese, e=English, j=Japanese)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--sample', action='store_true', help='Force create sample data and run analysis')

    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        show_help()
        return 0
    
    # Debug mode setup
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # If no input file provided, create sample data
    if not args.input_file or args.sample:
        print("🎯 No input file provided. Creating sample data...")
        try:
            from data_sample import create_sample_excel
            sample_file = create_sample_excel()
            args.input_file = str(sample_file)
            print(f"✅ Sample data created: {args.input_file}")
        except Exception as e:
            print(f"❌ Failed to create sample data: {str(e)}")
            return 1
    
    # Check input file (try output folder if not found)
    if not os.path.exists(args.input_file):
        # Try to find file in output folder
        output_path = Path(__file__).parent.parent / "output" / args.input_file
        if os.path.exists(output_path):
            args.input_file = str(output_path)
            print(f"📁 Found file in output folder: {args.input_file}")
        else:
            print(f"❌ Input file not found: {args.input_file}")
            print(f"💡 Make sure your Excel file is in the output/ folder")
            return 1
    
    # Check output directory
    if args.output and not os.path.exists(args.output):
        try:
            os.makedirs(args.output, exist_ok=True)
            print(f"📁 Created output directory: {args.output}")
        except Exception as e:
            print(f"❌ Failed to create output directory: {str(e)}")
            return 1
    
    try:
        # Run sentiment analysis system
        analyzer = MultiLanguageSentimentAnalyzer()
        success = analyzer.run_analysis(args.input_file, args.output, args.lang)
        
        if success:
            print("\n✅ Sentiment analysis completed successfully!")
            return 0
        else:
            print("\n❌ Error occurred during sentiment analysis.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1

def cli():
    """Entry point for the `sentiment-analysis` console script (see pyproject.toml)."""
    ensure_directories()
    sys.exit(main())

if __name__ == "__main__":
    cli()
