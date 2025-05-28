from datetime import datetime
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils.model_utils_classifier import Classifier
from model_utils.model_utils_pipeline import ClassificationPipeline
import matplotlib.pyplot as plt
import threading
from pathlib import Path
import time
import os
import matplotlib
matplotlib.use('Agg')  # This helps to ensure that the matplotlib does not pop oot but save the visuals
os.environ['MPLBACKEND'] = 'Agg'


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setting global state variables
classify = Classifier()
app_ready = False
model_training = False
startup_error = None
startup_message = "Starting up..."


plt.switch_backend('Agg')

def initialize_model():
    """Initialize model with training fallback"""
    global app_ready, model_training, startup_error, startup_message
    
    try:
        # Set thread-safe environment variables
        os.environ['MPLBACKEND'] = 'Agg'
        
        startup_message = "Loading existing model..."
        logger.info("Attempting to load existing model...")
        
        model_path = Path("models") / "saved_model.pkl"
        # Try to load existing model
        success = classify.load_model()
        
        if success:
            logger.info("Model loaded successfully from existing file")
            startup_message = "Model loaded successfully"
            app_ready = True
        else:
            logger.warning("Model file not found. Starting training process...")
            startup_message = "Model not found. Training new model..."
            model_training = True
            
            # Import and run training pipeline
            try:           
                
                logger.info("Starting model training. This may take several minutes...")
                startup_message = "Training model in progress. Please wait..."
                
                # Create pipeline and train model with thread-safe settings
                pipeline = ClassificationPipeline()
                
                # Ensure no GUI components are used during training
                training_success = pipeline.run_pipeline()
                
                if training_success:
                    logger.info("Model training completed successfully")
                    startup_message = "Training completed. Loading model..."
                    
                    import matplotlib.pyplot as plt

                    
                    # Clear any matplotlib figures to prevent memory issues after training pipeline
                    plt.close('all')
                    
                    model_path = Path("models") / "saved_model.pkl"
                    # Try to load the newly trained model
                    load_success = classify.load_model()
                    
                    if load_success:
                        logger.info("Newly trained model loaded successfully")
                        startup_message = "Model ready"
                        app_ready = True
                        model_training = False
                    else:
                        raise Exception("Failed to load newly trained model")
                else:
                    raise Exception("Model training failed")
                    
            except ImportError:
                error_msg = "ClassificationPipeline not available. Cannot train model."
                logger.error(error_msg)
                startup_error = error_msg
                startup_message = "Error: Training pipeline not available"
            except Exception as e:
                error_msg = f"Error during model training: {str(e)}"
                logger.error(error_msg, exc_info=True)
                startup_error = error_msg
                startup_message = f"Training failed: {str(e)}"
                
    except Exception as e:
        error_msg = f"Critical error during initialization: {str(e)}"
        logger.error(error_msg, exc_info=True)
        startup_error = error_msg
        startup_message = f"Startup failed: {str(e)}"
    
    finally:
        model_training = False
        # Clean up all remaining matplotlib resources
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass

# Middleware to check if app is ready
@app.before_request
def check_app_ready():
    """Check if the app is ready to serve requests"""
    # Allow status and health endpoints during startup
    allowed_during_startup = ['/status', '/health', '/']
    
    if request.endpoint and any(endpoint in request.path for endpoint in allowed_during_startup):
        return None
    
    if not app_ready:
        if startup_error:
            return jsonify({
                'error': 'Service unavailable',
                'message': startup_error,
                'status': 'error',
                'ready': False
            }), 503
        elif model_training:
            return jsonify({
                'error': 'Service starting',
                'message': 'Model is being trained. Please wait and try again.',
                'status': 'training',
                'ready': False,
                'estimated_wait': 'Several minutes'
            }), 503
        else:
            return jsonify({
                'error': 'Service starting',
                'message': startup_message,
                'status': 'initializing',
                'ready': False
            }), 503

# API Routes
@app.route('/', methods=['GET'])
def home():
    """API home endpoint with basic info"""
    return jsonify({
        'name': 'Legal Case Classification API',
        'description': 'Classify legal case reports into specific areas of law using BERT embeddings',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'status': '/status (GET)',
            'law_areas': '/law_areas (GET)',
            'model_info': '/model_info (GET)',
            'test': '/test (GET/POST)'
        },
        'status': 'running' if app_ready else 'starting',
        'ready': app_ready,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    return jsonify({
        'ready': app_ready,
        'training': model_training,
        'message': startup_message,
        'error': startup_error,
        'model_loaded': app_ready,
        'timestamp': datetime.now().isoformat()
    }), 200 if app_ready else 503

@app.route('/predict', methods=['POST'])
def predict():
    """Predict law area for a case report"""
    try:
        # Get model status
        model_status = classify.get_status()
        
        # Check if model is loaded
        if not model_status['model_loaded']:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The classification model is not available',
                'status': 'error'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Please provide JSON data with full_report field',
                'status': 'error'
            }), 400
        
        # Extract full_report
        full_report = data.get('full_report', str('')).strip()
        
        if not full_report:
            return jsonify({
                'error': 'Empty case report',
                'message': 'Please provide a non-empty full_report field',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = classify.predict(str(full_report))
        
        logger.info(f"Prediction made: {result['predicted_law_area']} (confidence: {result['confidence']})")
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e),
            'status': 'error'
        }), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred during prediction',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = classify.get_status() if app_ready else {'model_loaded': False}
    
    return jsonify({
        'status': 'healthy' if app_ready else 'starting',
        'ready': app_ready,
        'training': model_training,
        'model_loaded': model_status['model_loaded'],
        'message': startup_message,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200 if app_ready else 503

@app.route('/law_areas', methods=['GET'])
def get_law_areas():
    """Get supported law areas"""
    model_status = classify.get_status()
    return jsonify({
        'law_areas': model_status["law_areas"],
        'count': len(model_status["law_areas"]),
        'status': 'success'
    }), 200

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    model_status = classify.get_status()
    
    if not model_status['model_loaded']:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        return jsonify({
            'model_type': model_status['model_info']['model_type'],
            'classes': list(model_status['model_info']['classes']),
            'num_classes': model_status['model_info']['n_classes'],
            'model_path': model_status['model_info']['file_path'],
            'loaded_at': model_status['model_info']['loaded_at'],
            'status': 'success'
        }), 200
    except Exception as e:
        return jsonify({
            'error': 'Error getting model info',
            'message': str(e),
            'status': 'error'
        }), 500

# Open the text file in read mode and store its content as a string
try:
    file_path = Path("src") / "data" / "nlp.txtv"
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    content = "Sample text not found."
    logger.warning("Sample text file not found: ../src/data/nlp.txt")

# Test endpoint for development
@app.route('/test', methods=['GET', 'POST'])
def test():
    """Test endpoint with sample prediction"""
    if request.method == 'GET':
        return jsonify({
            'message': 'Test endpoint - use POST to test with sample data',
            'sample_data': {
                'full_report': content[:200] + "..." if len(content) > 200 else content
            }
        })
    
    # POST request - test with sample or provided data
    data = request.get_json() or {}
    
    sample_case = data.get('full_report', content)
    model_status = classify.get_status()
    
    try:
        if not model_status['model_loaded']:
            return jsonify({
                'error': 'Model not loaded for testing',
                'status': 'error'
            }), 500
        
        result = classify.predict(sample_case)
        result['test_mode'] = True
        result['sample_text'] = sample_case[:len(sample_case) - 50] + "..." if len(sample_case) > 50 else sample_case
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Test prediction failed',
            'message': str(e),
            'status': 'error'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist',
        'status': 'error'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The requested method is not allowed for this endpoint',
        'status': 'error'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    print("Starting Classification API...")
    print("="*50)
    
    # Start model initialization in a separate thread
    init_thread = threading.Thread(target=initialize_model)
    init_thread.daemon = True
    init_thread.start()
    
    print("Model initialization started in background...")
    print("API will be available at: http://localhost:5000")
    print("Status endpoint: http://localhost:5000/status")
    print("Health endpoint: http://localhost:5000/health")
    print("")
    print("Note: API endpoints will return 503 until model is ready")
    print("Use /status endpoint to check initialization progress")
    print("="*50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent threading issues
    )