# agents/cache.py
import sqlite3
import json
import hashlib
import time
import os
import numpy as np

class SQLiteCache:
    """
    SQLite-based persistent cache to store RAG responses and embedding representations
    to minimize LLM API cost, avoid re-computation, and reduce query latency.
    """
    def __init__(self, db_path="outputs/cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for full pipeline responses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT,
                    created_at REAL
                )
            """)
            # Table for query embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    embedding_blob BLOB,
                    created_at REAL
                )
            """)
            conn.commit()

    def _generate_key(self, *args, **kwargs) -> str:
        # Create a unique MD5 hash for the combination of inputs
        serialized = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def get_pipeline_response(self, pdf_path: str, query: str, **kwargs) -> dict:
        """
        Check and return cached pipeline output if it exists.
        Uses PDF content hash + query parameters for cache key validation.
        """
        pdf_hash = ""
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_key = self._generate_key(pdf_hash, query, kwargs)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM pipeline_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set_pipeline_response(self, pdf_path: str, query: str, response: dict, **kwargs):
        """
        Save pipeline response into the SQLite database.
        """
        pdf_hash = ""
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_key = self._generate_key(pdf_hash, query, kwargs)
        response_json = json.dumps(response, ensure_ascii=False)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO pipeline_cache (cache_key, response_json, created_at) VALUES (?, ?, ?)",
                (cache_key, response_json, time.time())
            )
            conn.commit()

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieve query embedding from cache if it exists.
        """
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_blob FROM embedding_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                # Load numpy array from blob
                return np.frombuffer(row[0], dtype=np.float32)
        return None

    def set_embedding(self, text: str, embedding: np.ndarray):
        """
        Store query embedding in the SQLite cache.
        """
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        embedding_blob = embedding.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO embedding_cache (cache_key, embedding_blob, created_at) VALUES (?, ?, ?)",
                (cache_key, embedding_blob, time.time())
            )
            conn.commit()

    def clear(self):
        """
        Clear all cached elements.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pipeline_cache")
            cursor.execute("DELETE FROM embedding_cache")
            conn.commit()
