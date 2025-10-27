# app/services/session_store.py
import os
import json
import redis
from typing import Optional, Dict, Any

class SessionStore:
    def __init__(self):
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', 6379))
        db = int(os.getenv('REDIS_DB', 0))
        password = os.getenv('REDIS_PASSWORD', None)
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            retry_on_timeout=True,
            health_check_interval=30
        )
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {e}")

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        key = f"session:{user_id}"
        data = self.redis_client.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None

    def set(self, user_id: str, sess: Dict[str, Any], ttl: int = 86400) -> bool:
        key = f"session:{user_id}"
        try:
            self.redis_client.set(key, json.dumps(sess))
            self.redis_client.expire(key, ttl)  # Default 24 hours
            return True
        except Exception:
            return False

    def delete(self, user_id: str) -> bool:
        key = f"session:{user_id}"
        try:
            return self.redis_client.delete(key) > 0
        except Exception:
            return False

    def close(self):
        self.redis_client.close()