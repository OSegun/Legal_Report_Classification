from typing import List
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTEmbedder:
    
    """
    Generate BERT embeddings for the text
    
    """

    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        logger.info(f"Loading BERT model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        
        """
        Generate BERT embeddings for a list of texts
        
        """
        
        self.model.eval()
        embeddings = []

        logger.info(f"Generating BERT embeddings for {len(texts)} texts...")
        logger.info(f"Batch size: {batch_size}, Max length: {self.max_length}")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

            # Progress update
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

        embeddings_array = np.vstack(embeddings)
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        return embeddings_array