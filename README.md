# Legal Case Classification System

A comprehensive NLP pipeline for classifying legal case reports into specific areas of law using BERT embeddings and machine learning.

## Law Areas Supported

- **Criminal Law and Procedure**
- **Civil Procedure**
- **Enforcement of Fundamental Rights**
- **Company Law**

## Project Structure

```
legal-case-classification/
├── notebooks  
|   ├─ code.ipynb                   # Data preprocessing and model training
├── app                            
|   ├─ api.py
├── model_utils
|   ├─ model_utils_classifier             # Trained model (generated after training)
|   ├─ model_utils_pipeline
├── src
|   ├─ data             # Trained model (generated after training)
|   ├─ embedding.py                        # Flask API deployment
|   ├─ preprocess.py
|   ├─ training.py
├── models
|   ├─ saved_model.pkl              # Trained model (generated after training)
|   ├─ saved_preprocessor.pkl       # Text preprocessor (generated after training)     
├── 
├── README.md 
└── requirements.txt                   # Python dependencies 
```

## Installation

1. **Clone or download the project files**

2. **Create a virtual environment:**
```bash
python -m venv legal_nlp_env
source legal_nlp_env/bin/activate  # On Windows: legal_nlp_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing and Model Training

1. **Prepare your dataset:**
   - CSV file should contain columns: `case_title`, `suitno`, `introduction`, `facts`, `issues`, `decision`, `full_report`
   - Place your CSV file in the src/data directory

2. **Open the Jupyter notebook:**
```bash
jupyter notebook code/notebook.ipynb
```

3. **Run the entire pipeline:**
```python
# bash
python -m model_utils_pipeline.py
```

This will:
- Clean and preprocess your data
- Extract law areas from the introduction text
- Generate BERT embeddings
- Train multiple classification models (Logistic Regression, Random Forest, SVM)
- Evaluate model performance
- Save the best model as `saved_model.pkl`
- Save the preprocessor as `saved_preprocessor.pkl`

### Step 2: Deploy the API

# Classification API Documentation

## Overview
The Classification API is a Flask-based service that classifies legal case reports into specific areas of law using BERT embeddings. The API provides endpoints for prediction, health checking, and model information.

**Base URL:** `http://localhost:5000` (when running locally)

## Getting Started

### Prerequisites
- Python 3.11+
- Required dependencies (installed via uv)
- Model files or training data

### Running the API
```bash
python -m app.api
```

The API will start on `http://localhost:5000` and begin initializing the model in the background.

## API Behavior

### Startup Process
The API follows a two-phase startup:
1. **Initialization Phase**: The API loads an existing model or trains a new one if the model is not found
2. **Ready Phase**: All endpoints become available for use

During initialization, most endpoints will return HTTP 503 with status information. Only `/status`, `/health`, and `/` are available during startup.

### Response Format
All responses are in JSON format with consistent structure:
```json
{
    "status": "success|error|training|initializing",
    "message": "Human readable message",
    "data": "Response data (varies by endpoint)"
}
```

## Endpoints

### 1. Home / Root
**GET** `/`

Returns basic API information and available endpoints.

**Response:**
```json
{
    "name": "Classification API",
    "description": "Classify legal case reports into specific areas of law using BERT embeddings",
    "endpoints": {
        "predict": "/predict (POST)",
        "health": "/health (GET)",
        "status": "/status (GET)",
        "law_areas": "/law_areas (GET)",
        "model_info": "/model_info (GET)",
        "test": "/test (GET/POST)"
    },
    "status": "running",
    "ready": true,
    "timestamp": "2025-05-27T10:30:00.000Z"
}
```

### 2. Health Check
**GET** `/health`

Returns the health status of the API and model.

**Response:**
```json
{
    "status": "healthy",
    "ready": true,
    "training": false,
    "model_loaded": true,
    "message": "Model ready",
    "timestamp": "2025-05-27T10:30:00.000Z",
    "version": "1.0.0"
}
```

**Status Codes:**
- `200`: API is ready and healthy
- `503`: API is starting up or has issues

### 3. Status
**GET** `/status`

Provides detailed status information about the API initialization process.

**Response:**
```json
{
    "ready": true,
    "training": false,
    "message": "Model ready",
    "error": null,
    "model_loaded": true,
    "timestamp": "2025-05-27T10:30:00.000Z"
}
```

**Status Codes:**
- `200`: API is ready
- `503`: API is still initializing

### 4. Predict (Main Classification Endpoint)
**POST** `/predict`

Classifies a legal case report into a specific area of law.

**Request Body:**
```json
{
    "full_report": "Your legal case report text here..."
}
```

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "full_report": "This is a contract dispute involving breach of agreement between two parties regarding delivery of goods..."
  }'
```

**Response:**
```json
{
    "predicted_law_area": "Contract Law",
    "confidence": 0.85,
    "all_predictions": {
        "Contract Law": 0.85,
        "Tort Law": 0.10,
        "Criminal Law": 0.05
    },
    "text_length": 156,
    "timestamp": "2025-05-27T10:30:00.000Z"
}
```

**Status Codes:**
- `200`: Successful prediction
- `400`: Invalid request (missing or empty full_report)
- `500`: Model not loaded or internal error
- `503`: API not ready

**Error Response Example:**
```json
{
    "error": "Empty case report",
    "message": "Please provide a non-empty full_report field",
    "status": "error"
}
```

### 5. Law Areas
**GET** `/law_areas`

Returns the list of supported law areas that the model can classify.

**Response:**
```json
{
    "law_areas": [
"Criminal Law and Procedure",
"Civil Procedure",
"Enforcement of Fundamental Rights",
"Company Law"
    ],
    "count": 5,
    "status": "success"
}
```

### 6. Model Information
**GET** `/model_info`

Returns detailed information about the loaded model.

**Response:**
```json
{
    "model_type": "BERT Classifier",
    "classes": [
"Criminal Law and Procedure",
"Civil Procedure",
"Enforcement of Fundamental Rights",
"Company Law"
    ],
    "num_classes": 4,
    "model_path": "/app/models/saved_model.pkl",
    "loaded_at": "2025-05-27T10:25:00.000Z",
    "status": "success"
}
```

**Status Codes:**
- `200`: Model information retrieved successfully
- `500`: Model not loaded

### 7. Test Endpoint
**GET** `/test`

Returns information about the test endpoint and sample data.

**Response:**
```json
{
    "message": "Test endpoint - use POST to test with sample data",
    "sample_data": {
        "full_report": "Sample legal case text..."
    }
}
```

**POST** `/test`

Tests the classification with sample data or provided data.

**Request Body (Optional):**
```json
{
    "full_report": "Optional custom text to test with"
}
```

**Response:**
```json
{
    "predicted_law_area": "Criminal Law and Procedure",
    "confidence": 0.75,
    "all_predictions": {...},
    "test_mode": true,
    "sample_text": "Sample case text used for testing...",
    "timestamp": "2024-05-27T10:30:00.000Z"
}
```

## Error Handling

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Endpoint not found
- `405`: Method not allowed
- `500`: Internal server error
- `503`: Service unavailable (during startup)

### Error Response Format
```json
{
    "error": "Error type",
    "message": "Detailed error message",
    "status": "error"
}
```

## Usage Examples

### Python Requests
```python
import requests
import json

# Check if API is ready
response = requests.get('http://localhost:5000/health')
print(response.json())

# Make a prediction
data = {
    "full_report": "This case involves a dispute over intellectual property rights..."
}
response = requests.post(
    'http://localhost:5000/predict',
    headers={'Content-Type': 'application/json'},
    json=data
)
result = response.json()
print(f"Prediction: {result['predicted_law_area']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript/Fetch
```javascript
// Check API status
fetch('http://localhost:5000/status')
  .then(response => response.json())
  .then(data => console.log('Status:', data));

// Make prediction
const predictionData = {
  full_report: "This case involves employment law issues..."
};

fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(predictionData)
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.predicted_law_area);
  console.log('Confidence:', data.confidence);
})
.catch(error => console.error('Error:', error));
```

### cURL Examples
```bash
# Check health
curl http://localhost:5000/health

# Get law areas
curl http://localhost:5000/law_areas

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"full_report": "Your legal case text here"}'

# Test with sample data
curl -X POST http://localhost:5000/test
```

## Ensure the following:

### 1. Check API Readiness
Always check the `/health` or `/status` endpoint before making predictions to ensure the model is loaded.

### 2. Handle Startup Period
The API may take several minutes to initialize, especially if training a new model. Implement retry logic with exponential backoff.

### 3. Input Validation
Ensure the `full_report` field contains meaningful legal case text for best classification results.

### 4. Error Handling
Always handle HTTP error status codes and parse error messages from the response.


## Troubleshooting

### Common Issues

**API Returns 503 "Service starting"**
- The model is still loading or training
- Check `/status` endpoint for progress
- Wait for initialization to complete

**Empty or Invalid Predictions**
- Ensure `full_report` contains sufficient legal case text
- Check that the text is relevant to the trained law areas

**Model Not Found Errors**
- The API will attempt to train a new model if none exists
- Ensure training data is available
- Check logs for training progress

**Connection Refused**
- Ensure the API is running on the correct port (5000)
- Check if the process started successfully
- Verify no firewall blocking the connection

## Development Notes

- The API runs in debug mode when started directly
- Model initialization happens in a background thread
- CORS is enabled for all routes
- Matplotlib backend is set to 'Agg' for headless operation

### Using Docker

Pull and run image from dockerhub:
```bash
docker pull segunodusina/legal-classifier-flask-api:latest
docker run -p 5000:5000 segunodusina/legal-classifier-flask-api:latest
```

## Troubleshooting

### Common Issues

1. **Model not found error:**
   - Ensure you've run the training notebook first
   - Check that `saved_model.pkl` exists in the project directory

2. **Out of memory during training:**
   - Reduce BERT batch size in the notebook
   - Use a smaller dataset for initial testing

3. **Slow predictions:**
   - The first prediction may be slow due to BERT model loading
   - Subsequent predictions should be faster

4. **Import errors:**
   - Ensure all dependencies are installed: `uv pip install -r requirements.txt`
   - Check Python version compatibility (3.8+ recommended)

### Performance Tips

1. **For faster inference:**
   - Use GPU if available (CUDA-enabled PyTorch)
   - Consider model quantization for production

2. **For better accuracy:**
   - Increase training data size
   - Fine-tune BERT on legal domain data
   - Experiment with different text preprocessing approaches

## File Descriptions

- **`code.ipynb`**: Complete data preprocessing and model training pipeline
- **`api.py`**: Flask API application for model deployment
- **`requirements.txt`**: Python package dependencies
- **`saved_model.pkl`**: Saved trained model (generated after training)
- **`saved_preprocessor.pkl`**: Saved text preprocessor (generated after training)

## License

This project is provided as-is for poc purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify that model files exist before running the API

---

**Note**: This system is designed for poc purposes. For production legal applications, additional validation and testing should be performed.