# app/services/embeddings.py
import os
import time
import logging
from functools import lru_cache
from typing import List

from ..config import AppConfig

try:
    from openai import OpenAI
except ImportError:
    # fallback if old package is installed
    import openai
    class _Wrapper:
        def __init__(self): self._c = openai
        def embeddings(self): return self._c.Embedding
    OpenAI = _Wrapper  # type: ignore

# Choose your embedding model:
# - "text-embedding-3-small"  -> 1536 dims, cheaper
# - "text-embedding-3-large"  -> 3072 dims, better quality
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))

log = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _client():
    api_key = AppConfig.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key) if hasattr(OpenAI, "api_key") else OpenAI()

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
        
    client = _client()
    backoff = 1.0
    
    for attempt in range(5):
        try:
            log.debug(f"Sending batch of {len(texts)} texts to OpenAI for embedding")
            resp = client.embeddings.create(
                model=OPENAI_EMBED_MODEL, 
                input=texts
            )
            
            if not hasattr(resp, 'data') or not resp.data:
                raise ValueError("No data in OpenAI response")
                
            vectors = [d.embedding for d in resp.data]
            log.debug(f"Successfully generated {len(vectors)} embedding vectors")
            if vectors and len(vectors[0]) > 0:
                log.debug(f"Vector dimension: {len(vectors[0])}")
                
            return vectors
            
        except Exception as e:
            if attempt == 4:
                log.error(f"Failed to generate embeddings after 5 attempts: {str(e)}", exc_info=True)
                raise
                
            wait_time = min(backoff * (2 ** attempt), 30)  # Exponential backoff with max 30s
            log.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    return []  # Should never reach here due to raise in the loop

def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Convert list of texts into OpenAI embeddings (list of lists of floats).
    """
    texts = texts or []
    if not texts:
        return []
        
    log.info(f"Generating embeddings for {len(texts)} texts")
    out: List[List[float]] = []
    
    try:
        for i, batch in enumerate(_chunk(texts, BATCH_SIZE)):
            log.debug(f"Processing batch {i+1} of {len(list(_chunk(texts, BATCH_SIZE)))}")
            vectors = _embed_batch(batch)
            out.extend(vectors)
            
        log.info(f"Successfully generated {len(out)} embeddings")
        return out
        
    except Exception as e:
        log.error(f"Error in embed_text: {str(e)}", exc_info=True)
        raise
