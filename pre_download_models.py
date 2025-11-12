#!/usr/bin/env python3
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def download_onnx_model():
    try:
        log.info("Checking for ONNX model...")
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        
        # This will download the model if it doesn't exist
        model = ONNXMiniLM_L6_V2()
        
        # Test the model to ensure it's working
        test_embedding = model(["test"])
        if test_embedding and len(test_embedding) > 0:
            log.info("✓ ONNX model is ready to use")
            return True
            
    except Exception as e:
        log.error(f"Failed to download ONNX model: {e}")
        return False

if __name__ == "__main__":
    print("Pre-downloading required models...")
    success = download_onnx_model()
    if success:
        print("✓ Model download completed successfully")
    else:
        print("✗ Failed to download model. Check logs for details.")
