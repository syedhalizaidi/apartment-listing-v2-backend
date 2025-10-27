#!/usr/bin/env python3
import os
import shutil
import logging
from app.services.vector_store import ListingStore
from app.config import AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database():
    # Set the ChromaDB directory
    chroma_dir = os.path.abspath(AppConfig.CHROMA_DIR or "./chroma")
    
    # Remove the existing chroma directory if it exists
    if os.path.exists(chroma_dir):
        logger.info(f"Removing existing Chroma directory: {chroma_dir}")
        shutil.rmtree(chroma_dir)
    else:
        logger.info(f"Chroma directory does not exist, will create new one at: {chroma_dir}")
    
    # Ensure the directory exists
    os.makedirs(chroma_dir, exist_ok=True)
    
    # Create a new store which will create a fresh collection
    logger.info("Creating new ChromaDB collection...")
    try:
        store = ListingStore(force_recreate=True)
        logger.info(f"Successfully created new collection: {store.collection.name}")
        logger.info(f"Collection metadata: {store.collection.metadata}")
        return True
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        return False

if __name__ == "__main__":
    if reset_database():
        logger.info("Database reset completed successfully!")
    else:
        logger.error("Database reset failed!")
