# app/services/embedding_service.py
# Version 2: Reads config from environment variables

import os # <-- Add import for os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import logging
from typing import Optional, Tuple

# --- Removed EmbeddingModel and db imports ---
# from ..models import EmbeddingModel # No longer needed
# from .. import db # No longer needed

# Import OpenAI client instance and possible error types
# Ensure openai_service correctly initializes the client from .env
from .openai_service import client as openai_client, AuthenticationError, APIConnectionError, RateLimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local SentenceTransformer model cache
local_model_cache = {}

# --- Define Default Embedding Configuration ---
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "doubao-embedding-large-text")
DEFAULT_EMBEDDING_INVOCATION_METHOD = os.getenv("OPENAI_EMBEDDING_INVOCATION_METHOD", "remote_api")
# --- Default Config End ---


# --- Modified: Read config directly from environment variables ---
def get_active_embedding_model_info() -> Tuple[str, str]:
    """
    Gets the Embedding Model configuration from environment variables.

    Reads OPENAI_EMBEDDING_MODEL_NAME and OPENAI_EMBEDDING_INVOCATION_METHOD
    from the environment. Falls back to predefined defaults if not set.

    Returns:
        Tuple[str, str]: Returns (model_name, invocation_method) tuple.
    """
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)
    invocation_method = os.getenv("OPENAI_EMBEDDING_INVOCATION_METHOD", DEFAULT_EMBEDDING_INVOCATION_METHOD)

    # Log the resolved configuration
    # Avoid logging sensitive parts if model_name could contain secrets
    logger.info(f"Using Embedding Model (from env or default): {model_name} (Method: {invocation_method})")

    # Basic validation (optional but recommended)
    if invocation_method not in ['local', 'remote_api']:
        logger.warning(f"Unsupported invocation method '{invocation_method}' found in env. Falling back to default '{DEFAULT_EMBEDDING_INVOCATION_METHOD}'.")
        invocation_method = DEFAULT_EMBEDDING_INVOCATION_METHOD

    if not model_name:
        logger.error("Embedding model name is empty even after checking env and defaults. Cannot proceed.")
        # Handle this case appropriately, maybe raise an error or return a specific tuple
        # For now, let's return defaults again, but log error
        return DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_EMBEDDING_INVOCATION_METHOD

    return model_name, invocation_method

def _load_local_model(model_name: str) -> Optional[SentenceTransformer]:
    """Loads a local SentenceTransformer model, using cache."""
    global local_model_cache
    if model_name in local_model_cache:
        logger.debug(f"Using cached local model: {model_name}")
        return local_model_cache[model_name]
    try:
        # Check if SentenceTransformer is available
        if 'SentenceTransformer' not in globals():
             logger.error("SentenceTransformer library not imported or available. Cannot load local model.")
             return None
        logger.info(f"Loading local SentenceTransformer model: {model_name}...")
        model = SentenceTransformer(model_name)
        local_model_cache[model_name] = model
        logger.info(f"Successfully loaded and cached local model: {model_name}")
        return model
    except NameError: # Specifically catch if SentenceTransformer itself is not defined
         logger.error("SentenceTransformer library is required for local models but not loaded.", exc_info=False)
         return None
    except Exception as e:
        logger.error(f"Error loading local SentenceTransformer model '{model_name}': {e}", exc_info=True)
        return None

def generate_embedding(text: str) -> Optional[bytes]:
    """
    Generates text embedding based on configuration from environment variables.
    """
    if not text or not isinstance(text, str):
        logger.error("Invalid input text provided for embedding generation.")
        return None

    # Get config directly from env vars via the helper function
    model_name, invocation_method = get_active_embedding_model_info()

    # --- Added check from previous version ---
    if not model_name or not invocation_method:
        logger.error("Could not determine active embedding model configuration. Cannot generate embedding.")
        return None
    # --- Check end ---

    logger.info(f"Generating embedding using model '{model_name}' via '{invocation_method}' method.")
    vector = None

    if invocation_method == 'local':
        model = _load_local_model(model_name)
        if model:
            try:
                vector = model.encode(text)
                logger.info(f"Successfully generated embedding locally using {model_name}.")
            except Exception as e:
                logger.error(f"Error generating embedding with local model {model_name}: {e}", exc_info=True)
        else:
            logger.error(f"Local model {model_name} could not be loaded. Cannot generate embedding.")

    elif invocation_method == 'remote_api':
        # Ensure the client from openai_service is used and initialized
        if not openai_client:
            logger.error("OpenAI client (imported from openai_service) is not initialized. Cannot generate remote embedding.")
            return None
        try:
            response = openai_client.embeddings.create(
                model=model_name, # Use model name from env vars
                input=[text]
            )
            if response.data and response.data[0].embedding:
                vector = np.array(response.data[0].embedding)
                logger.info(f"Successfully generated embedding via remote API using {model_name}.")
            else:
                # Log the actual response if possible (be careful about sensitive data)
                logger.error(f"Remote API call for embedding using {model_name} returned unexpected data. Response: {response}")
        except (AuthenticationError, APIConnectionError, RateLimitError) as e:
             # Log the specific error type and message
             logger.error(f"API error during remote embedding generation ({type(e).__name__}) using {model_name}: {e}", exc_info=False) # Set exc_info=False for API errors unless debugging needed
        except Exception as e:
            logger.error(f"Unexpected error during remote embedding generation using {model_name}: {e}", exc_info=True)

    else:
        logger.error(f"Unsupported invocation method configured: {invocation_method}")

    # Serialize the vector
    if vector is not None:
        try:
            # Ensure vector is numpy array before pickling
            if not isinstance(vector, np.ndarray):
                 vector = np.array(vector)
            serialized_vector = pickle.dumps(vector)
            return serialized_vector
        except (pickle.PicklingError, TypeError) as e:
            logger.error(f"Error serializing the generated vector: {e}", exc_info=True)
            return None
    else:
        logger.error("Embedding generation failed (vector is None).")
        return None

def deserialize_embedding(serialized_embedding: bytes) -> Optional[np.ndarray]:
    """Deserializes stored binary data back into a NumPy vector."""
    if not serialized_embedding: return None
    try:
        vector = pickle.loads(serialized_embedding)
        # Add type check after deserialization
        return vector if isinstance(vector, np.ndarray) else None
    except (pickle.UnpicklingError, TypeError, EOFError) as e: # Catch more specific errors
        logger.error(f"Error deserializing embedding: {e}", exc_info=True)
        return None
    except Exception as e: # Catch any other unexpected errors
         logger.error(f"Unexpected error during deserialization: {e}", exc_info=True)
         return None


# --- Helper functions for distance calculation remain unchanged ---
def calculate_cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine distance between two NumPy vectors."""
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        logger.error("Inputs must be NumPy arrays for distance calculation.")
        return float('inf')
    if vec1.ndim == 1: vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1: vec2 = vec2.reshape(1, -1)
    # Add shape check
    if vec1.shape[1] != vec2.shape[1]:
        logger.error(f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}")
        return float('inf')
    try:
        distance = cdist(vec1, vec2, 'cosine')[0, 0]
        # Handle potential NaN result from cdist if vectors are zero vectors
        return float(distance) if not np.isnan(distance) else 1.0 # Cosine distance of 1 for zero vectors vs others
    except Exception as e:
        logger.error(f"Error calculating cosine distance: {e}", exc_info=True)
        return float('inf')

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two NumPy vectors."""
    distance = calculate_cosine_distance(vec1, vec2)
    if distance == float('inf'): return -1.0 # Indicate error or incompatibility
    # Clamp similarity between -1 and 1
    similarity = 1.0 - distance
    return max(-1.0, min(1.0, similarity))