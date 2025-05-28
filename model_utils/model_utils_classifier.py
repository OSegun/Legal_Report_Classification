import os
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
from .model_utils_pipeline import ClassificationPipeline
from src.embedding import BERTEmbedder
from src.preprocess import DataProcessor



class LoggerMixin:
    """
    Mixin, a helper class to work alongside other classes and 
    
    provide consistent logging across components
    
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for the class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger


class ModelManager(LoggerMixin):
    """
    Handle model loading, validation, and training a new model 
        
        using ClassificationPipeline with configuration, if model is not found
        
        in path.
    """
    
    def __init__(self, auto_train=True, pipeline_config=None):
        self.model = None
        self.label_encoder = None
        self.model_metadata = {}
        self._is_loaded = False
        self.auto_train = auto_train
        self.pipeline_config = pipeline_config or {}
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self._is_loaded and self.model is not None and self.label_encoder is not None
    
    model_path = Path("models") / "saved_model.pkl"
    def load_model(self, model_path: str = model_path) -> bool:
        """
        Load trained model from file. If model not found and auto_train is True, 
        run ClassificationPipeline to train one.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                
                if self.auto_train:
                    self.logger.info("Auto-training enabled. Attempting to train a new model...")
                    
                    if self._train_new_model(model_path):
                        self.logger.info("New model trained successfully. Proceeding to load it.")
                    else:
                        self.logger.error("Failed to train new model")
                        return False
                else:
                    self.logger.error("Auto-training disabled. Cannot proceed without existing model.")
                    return False
            
            self.logger.info(f"Loading model from: {model_path}")
            
            # Load model data
            model_data = joblib.load(model_path)
            
            # Validate model data structure
            if not self._validate_model_data(model_data):
                return False
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.model_metadata = {
                'model_type': model_data.get('model_type', 'Unknown'),
                'classes': list(self.label_encoder.classes_),
                'n_classes': len(self.label_encoder.classes_),
                'loaded_at': datetime.now().isoformat(),
                'file_path': str(model_path),
                'auto_trained': not model_path.exists() and self.auto_train
            }
            
            self._is_loaded = True
            
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Model type: {self.model_metadata['model_type']}")
            self.logger.info(f"Classes ({self.model_metadata['n_classes']}): {self.model_metadata['classes']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self._reset_model()
            return False
    
    def _train_new_model(self, model_path: Path) -> bool:
        """
        Training models with configurations.
        
        Args:
            model_path: Path where the new model should be saved
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            
            
            self.logger.info("Initializing ClassificationPipeline for model training...")
            
            # Create the pipeline instance with configuration
            pipeline = ClassificationPipeline(**self.pipeline_config)
            
            # Ensure the directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run the training pipeline
            success = pipeline.run_pipeline()
            
            if success and model_path.exists():
                self.logger.info(f"Model training completed successfully. Model saved to: {model_path}")
                return True
            else:
                self.logger.error("Model training completed but model file was not created")
                return False
                
        except ImportError as e:
            self.logger.error(f"Could not import ClassificationPipeline: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}", exc_info=True)
            return False
    
    def _validate_model_data(self, model_data: Dict) -> bool:
        """Validate loaded model data structure"""
        required_keys = ['model', 'label_encoder']
        
        for key in required_keys:
            if key not in model_data:
                self.logger.error(f"Missing required key in model data: {key}")
                return False
        
        if not hasattr(model_data['model'], 'predict'):
            self.logger.error("Loaded model does not have predict method")
            return False
        
        if not hasattr(model_data['label_encoder'], 'inverse_transform'):
            self.logger.error("Loaded label encoder does not have inverse_transform method")
            return False
        
        return True
    
    def _reset_model(self):
        """Reset model state"""
        self.model = None
        self.label_encoder = None
        self.model_metadata = {}
        self._is_loaded = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        return self.model_metadata.copy()


class TextProcessor(LoggerMixin):
    """Handles text preprocessing and embedding generation"""
    
    def __init__(self):
        self.preprocessor = None
        self.embedder = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize preprocessing and embedding components"""
        try:
            self.preprocessor = DataProcessor()
            self.embedder = BERTEmbedder()
            self.logger.info("Text processing components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize text processing components: {str(e)}")
            raise
    
    def process_text(self, text: str) -> Tuple[str, np.ndarray]:
        """
        Process text and generate embeddings
        
        Args:
            text: Raw input text
            
        Returns:
            Tuple of (cleaned_text, embeddings)
        """
        if not text or text.strip() == "":
            raise ValueError("Empty text provided")
        
        try:
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            self.logger.debug(f"Text cleaned: {len(text)} -> {len(cleaned_text)} characters")
            
            if len(cleaned_text) < 10:
                raise ValueError("Text too short after cleaning (< 10 characters)")
            
            # Generate embeddings
            embeddings = self.embedder.get_embeddings([cleaned_text])
            self.logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            
            return cleaned_text, embeddings
            
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise


class PredictionEngine(LoggerMixin):
    """Handles prediction logic and result formatting"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def predict(self, embeddings: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """
        Make predictions using the loaded model
        
        Args:
            embeddings: Input embeddings
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing prediction results
        """
        if not self.model_manager.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Make prediction
            prediction = self.model_manager.model.predict(embeddings)[0]
            probabilities = self.model_manager.model.predict_proba(embeddings)[0]
            
            # Get predicted class name
            predicted_class = self.model_manager.label_encoder.inverse_transform([prediction])[0]
            confidence = float(probabilities[prediction])
            
            self.logger.info(f"Prediction made: {predicted_class} (confidence: {confidence:.4f})")
            
            # Get top k predictions
            top_predictions = self._get_top_predictions(probabilities, top_k)
            
            return {
                'predicted_law_area': predicted_class,
                'confidence': round(confidence, 4),
                'top_predictions': top_predictions,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'model_info': self.model_manager.get_model_info()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _get_top_predictions(self, probabilities: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Get top k predictions with probabilities"""
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.model_manager.label_encoder.inverse_transform([idx])[0]
            prob = float(probabilities[idx])
            top_predictions.append({
                'law_area': class_name,
                'confidence': round(prob, 4)
            })
        
        return top_predictions


class Classifier(LoggerMixin):
    """Main classifier for legal cases - orchestrates all components"""
    
    # Define law areas as class constant
    LAW_AREAS = [
        "Criminal Law and Procedure",
        "Civil Procedure", 
        "Enforcement of Fundamental Rights",
        "Company Law"
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_manager = ModelManager()
        self.text_processor = None
        self.prediction_engine = None
        self.model_metadata = self.model_manager.model_metadata
        
        # Initialize components
        self._initialize_components()
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _initialize_components(self):
        """Initialize all classifier components"""
        try:
            self.text_processor = TextProcessor()
            self.prediction_engine = PredictionEngine(self.model_manager)
            self.logger.info("Legal case classifier initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier components: {str(e)}")
            raise
        
        
    model_path = Path("models") / "saved_model.pkl"
    def load_model(self, model_path: str = model_path) -> bool:
        """
        Load the trained model
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            bool: True if model loaded successfully
        """
        success = self.model_manager.load_model(model_path)
        if success:
            self.logger.info("Classifier ready for predictions")
        else:
            self.logger.warning("Classifier not ready - model loading failed")
        return success
    
    def predict(self, full_report: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Predict law area for a legal case report
        
        Args:
            full_report: Full text of the legal case report
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing prediction results
        """
        if not self.is_ready():
            raise ValueError("Classifier not ready. Please load the model first.")
        
        try:
            self.logger.info("Starting prediction process")
            
            # Process text and generate embeddings
            cleaned_text, embeddings = self.text_processor.process_text(full_report)
            
            # Make prediction
            result = self.prediction_engine.predict(embeddings, top_k)
            
            # Add processing metadata
            result['processing_info'] = {
                'original_length': len(full_report),
                'cleaned_length': len(cleaned_text),
                'embedding_shape': list(embeddings.shape)
            }
            
            self.logger.info(f"Prediction completed successfully: {result['predicted_law_area']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction process failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def is_ready(self) -> bool:
        """Check if classifier is ready for predictions"""
        return (self.model_manager.is_loaded and 
                self.text_processor is not None and 
                self.prediction_engine is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the classifier"""
        return {
            'is_ready': self.is_ready(),
            'model_loaded': self.model_manager.is_loaded,
            'model_info': self.model_manager.get_model_info(),
            'law_areas': self.LAW_AREAS,
            'components': {
                'text_processor': self.text_processor is not None,
                'prediction_engine': self.prediction_engine is not None
            }
        }


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize classifier
    try:
        classifier = Classifier()
        status = classifier.get_status()
        
        
        # Load model
        success = classifier.load_model()#"../models/saved_model.pkl")
        
        if success:
            print(classifier.model_metadata)
            print("Model loaded successfully!")
            print(f"Status: {classifier.get_status()}")
            print(status['model_loaded'])
            
            # Example prediction (uncomment to test)
            # sample_text = "Your legal case text here..."
            # result = classifier.predict(sample_text)
            # print(f"Prediction result: {result}")
        else:
            print("Failed to load model")
            
    except Exception as e:
        logging.error(f"Application error: {str(e)}", exc_info=True)