import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from src.embedding import BERTEmbedder
from src.training import ClassifierTrainer
from src.preprocess import DataProcessor


class ClassificationPipeline:
    """
    An end-to-end pipeline for classification.
    """
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the Classification Pipeline
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path. If None, logs to console only.
        """
        self.preprocessor = DataProcessor()
        self.embedder = BERTEmbedder()
        self.trainer = None
        self.df_processed = None
        self.results = None
        
        # Setup logging
        self._setup_logging(log_level, log_file)
        self.logger.info("Classification Pipeline initialized")
    
    def _setup_logging(self, log_level: str, log_file: Optional[str]) -> None:
        """Setup logging configuration"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    
    data_path = Path("src") / "data" / "sample_200_rows.csv"
    def load_dataset(self, file_path: str = data_path) -> Optional[pd.DataFrame]:
        """
        Load the dataset
        
        Args:
            file_path: Path to the CSV dataset file
            
        Returns:
            DataFrame if successful, None if failed
        """
        try:
            self.logger.info(f"Loading dataset from: {file_path}")
            
            if not Path(file_path).exists():
                self.logger.error(f"Dataset file not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Dataset loaded successfully! Shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None
    
    def explore_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explore the dataset structure and content
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional exploration columns
        """
        
        self.logger.info("Starting dataset exploration")
        
        try:
            self.logger.info(f"Dataset Shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Missing values analysis
            missing_values = df.isnull().sum()
            self.logger.info("Missing Values Analysis:")
            for col, count in missing_values.items():
                if count > 0:
                    self.logger.warning(f"  {col}: {count} missing values")
            
            # Data types
            self.logger.debug("Data Types:")
            for col, dtype in df.dtypes.items():
                self.logger.debug(f"  {col}: {dtype}")
            
            # Text length 
            if 'full_report' in df.columns:
                df['report_length'] = df['full_report'].astype(str).str.len()
                length_stats = df['report_length'].describe()
                self.logger.info("Text Length Statistics:")
                self.logger.info(f"  Mean: {length_stats['mean']:.2f}")
                self.logger.info(f"  Min: {length_stats['min']:.0f}")
                self.logger.info(f"  Max: {length_stats['max']:.0f}")
                
            
            self.logger.info("Dataset exploration completed")
            return df
            
        except Exception as e:
            self.logger.error(f"Error during dataset exploration: {e}")
            return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame or None if failed
        """
        try:
            self.logger.info("Starting data preprocessing")
            df_processed = self.preprocessor.process_dataset(df)
            
            if df_processed is not None:
                self.logger.info(f"Data preprocessing completed. Shape: {df_processed.shape}")
                
                # Log preprocessing statistics
                if 'cleaned_report' in df_processed.columns:
                    avg_length = df_processed['cleaned_report'].str.len().mean()
                    self.logger.info(f"Average cleaned text length: {avg_length:.2f}")
            else:
                self.logger.error("Data preprocessing returned None")
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}")
            return None
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 4) -> Optional[Any]:
        """
        Generate BERT embeddings for the texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for embedding generation
            
        Returns:
            Embeddings array or None if failed
        """
        try:
            self.logger.info(f"Generating BERT embeddings for {len(texts)} texts")
            self.logger.info(f"Using batch size: {batch_size}")
            
            embeddings = self.embedder.get_embeddings(texts, batch_size=batch_size)
            
            if embeddings is not None:
                self.logger.info(f"Embeddings generated successfully. Shape: {embeddings.shape}")
            else:
                self.logger.error("Embedding generation returned None")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error during embedding generation: {e}")
            return None
    
    def train_models(self, df_processed: pd.DataFrame, embeddings: Any) -> Optional[Dict]:
        """
        Train classification models
        
        Args:
            df_processed: Processed DataFrame
            embeddings: BERT embeddings
            
        Returns:
            Training results dictionary or None if failed
        """
        try:
            self.logger.info("Starting model training")
            
            self.trainer = ClassifierTrainer()
            X, y = self.trainer.prepare_data(df_processed, embeddings)
            
            self.logger.info(f"Training data prepared. X shape: {X.shape}, y shape: {y.shape}")
            
            results = self.trainer.train_models(X, y)
            
            if results:
                self.logger.info("Model training completed successfully")
                for model_name, metrics in results.items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        self.logger.info(f"{model_name} accuracy: {metrics['accuracy']:.4f}")
            else:
                self.logger.warning("Model training completed but no results returned")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            return None
    
    def evaluate_models(self) -> bool:
        """
        Evaluate trained models
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.trainer is None:
                self.logger.error("No trainer available for evaluation")
                return False
            
            self.logger.info("Starting model evaluation")
            self.trainer.evaluate_models()
            self.logger.info("Model evaluation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            return False
        
        
    model_path = Path("models") / "saved_model.pkl"
    process_path = Path("models") / "saved_preprocessor.pkl"
    
    def save_models(self, model_path: str = model_path, 
                   preprocessor_path: str = process_path) -> bool:
        """
        Save trained models and preprocessor
        
        Args:
            model_path: Path to save the trained model
            preprocessor_path: Path to save the preprocessor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.trainer is None:
                self.logger.error("No trained model available to save")
                return False
            
            self.logger.info(f"Saving model to: {model_path}")
            self.trainer.save_model(model_path)
            
            self.logger.info(f"Saving preprocessor to: {preprocessor_path}")
            joblib.dump(self.preprocessor, preprocessor_path)
            
            self.logger.info("Models saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    def run_pipeline(self, csv_file_path: str=data_path, batch_size: int = 8) -> bool:
        """
        Run the complete training pipeline
        
        Args:
            csv_file_path: Path to the dataset CSV file
            batch_size: Batch size for embedding generation
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        self.logger.info("="*80)
        self.logger.info("STARTING CLASSIFICATION PIPELINE")
        self.logger.info("="*80)
        
        # Load and explore data
        df = self.load_dataset(csv_file_path)
        if df is None:
            self.logger.error("Pipeline failed: Could not load dataset")
            return False
        
        df = self.explore_dataset(df)
        
        # Preprocess data
        self.df_processed = self.preprocess_data(df)
        if self.df_processed is None:
            self.logger.error("Pipeline failed: Data preprocessing failed")
            return False
        
        # Generate embeddings
        texts = self.df_processed['cleaned_report'].tolist()
        embeddings = self.generate_embeddings(texts, batch_size)
        if embeddings is None:
            self.logger.error("Pipeline failed: Embedding generation failed")
            return False
        
        # Train models
        self.results = self.train_models(self.df_processed, embeddings)
        if self.results is None:
            self.logger.error("Pipeline failed: Model training failed")
            return False
        
        # Evaluate models
        if not self.evaluate_models():
            self.logger.warning("Model evaluation failed, but continuing...")
        
        # Save models
        if not self.save_models():
            self.logger.warning("Model saving failed, but pipeline completed")
        
        self.logger.info("="*80)
        self.logger.info("CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)
        
        return True
    
    def predict(self, text: str, model_path: str = model_path) -> Optional[Tuple[str, float]]:
        
        """
        Make prediction on new text
        
        Args:
            text: Input text for prediction
            model_path: Path to the saved model
            
        Returns:
            Tuple of (predicted_class, confidence) or None if failed
        """
        try:
            self.logger.info("Making prediction on new text")
            self.logger.debug(f"Input text preview: {text[:len(text) - 10]}...")
            
            # Load model
            if not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            model_data = joblib.load(model_path)
            model = model_data['model']
            label_encoder = model_data['label_encoder']
            
            self.logger.debug("Model loaded successfully")
            
            # Preprocess
            cleaned_text = self.preprocessor.clean_text(text)
            self.logger.debug(f"Text cleaned, length: {len(cleaned_text)}")
            
            # Generate embedding
            embedding = self.embedder.get_embeddings([cleaned_text])
            if embedding is None:
                self.logger.error("Failed to generate embedding for prediction")
                return None
            
            # Predict
            prediction = model.predict(embedding)[0]
            probabilities = model.predict_proba(embedding)[0]
            
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            self.logger.info(f"Prediction completed: {predicted_class} (confidence: {confidence:.4f})")
            
            return predicted_class, confidence
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline state and results
        
        Returns:
            Dictionary containing pipeline summary
        """
        summary = {
            'pipeline_initialized': True,
            'data_loaded': self.df_processed is not None,
            'model_trained': self.trainer is not None,
            'results_available': self.results is not None
        }
        
        if self.df_processed is not None:
            summary['dataset_shape'] = self.df_processed.shape
            summary['columns'] = list(self.df_processed.columns)
        
        if self.results is not None:
            summary['training_results'] = self.results
        
        self.logger.info("Pipeline summary generated")
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with logging
    pipeline = ClassificationPipeline(
        log_level="INFO",
        log_file="pipeline.log"
    )
    
    # Run complete pipeline
    success = pipeline.run_pipeline() #"data/sample_200_rows.csv")
    file_path = Path("src") / "data" / "nlp.txt"
    # Open a full report from a txt file
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        content = file.read()
        
    if success:
        # Make prediction
        result = pipeline.predict(content)
        
        if result:
            predicted_class, confidence = result
            print(f"Predicted Law Area: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        print("\nPipeline Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")