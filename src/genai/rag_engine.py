import numpy as np
import io
import os
import requests
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

class LocalVectorStore:
    """
    Local Vector Store database supporting dense vector search (online) 
    and TF-IDF token search (offline fallback).
    """
    def __init__(self, api_key: Optional[str] = None, provider: str = "gemini"):
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.provider = provider.lower() if provider else "gemini"
        self.chunks = []  # List of {"text": str, "source": str, "vector": np.ndarray}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def clear(self):
        """
        Clears the stored vector indices and text chunks.
        """
        self.chunks = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def get_chunks_count(self) -> int:
        return len(self.chunks)
        
    def add_document(self, filename: str, content_bytes: bytes):
        """
        Extracts text from TXT, PDF, or DOCX formats, chunks it, and adds to local collection.
        """
        ext = filename.split(".")[-1].lower()
        text = ""
        
        try:
            if ext == "txt":
                text = content_bytes.decode("utf-8", errors="ignore")
            elif ext == "pdf":
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content_bytes))
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            elif ext in ["docx", "doc"]:
                import docx
                doc = docx.Document(io.BytesIO(content_bytes))
                for p in doc.paragraphs:
                    text += p.text + "\n"
            else:
                text = content_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            # Safe fallback text decode
            text = content_bytes.decode("utf-8", errors="ignore")
            
        if not text.strip():
            return
            
        new_chunks = self._chunk_text(text, chunk_size=500, overlap=100)
        for chunk in new_chunks:
            self.chunks.append({
                "text": chunk,
                "source": filename,
                "vector": None
            })
            
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        chunks = []
        words = text.split()
        text_clean = " ".join(words)
        
        start = 0
        while start < len(text_clean):
            end = start + chunk_size
            chunks.append(text_clean[start:end])
            start += (chunk_size - overlap)
        return chunks
        
    def build_index(self):
        """
        Builds the search indices. Attempts dense vector embed first, falls back to sparse TF-IDF.
        """
        if not self.chunks:
            return
            
        if self.api_key:
            try:
                for chunk in self.chunks:
                    chunk["vector"] = self._get_embedding(chunk["text"])
                return
            except Exception:
                pass
                
        # TF-IDF Sparse Index Fallback (Offline)
        corpus = [c["text"] for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
    def _get_embedding(self, text: str) -> np.ndarray:
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
            payload = {"content": {"parts": [{"text": text}]}}
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                return np.array(res.json()["embedding"]["values"], dtype=np.float32)
        else: # openai
            url = "https://api.openai.com/v1/embeddings"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"model": "text-embedding-3-small", "input": text}
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                return np.array(res.json()["data"][0]["embedding"], dtype=np.float32)
                
        raise Exception("Failed to generate embedding")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant text chunks matching the search query.
        """
        if not self.chunks:
            return []
            
        # 1. Dense Vector Search
        if self.chunks[0]["vector"] is not None and self.api_key:
            try:
                query_vec = self._get_embedding(query)
                scores = []
                for chunk in self.chunks:
                    vec = chunk["vector"]
                    dot_prod = np.dot(query_vec, vec)
                    norm_q = np.linalg.norm(query_vec)
                    norm_v = np.linalg.norm(vec)
                    sim = dot_prod / (norm_q * norm_v) if norm_q > 0 and norm_v > 0 else 0.0
                    scores.append(sim)
                    
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    results.append({
                        "text": self.chunks[idx]["text"],
                        "source": self.chunks[idx]["source"],
                        "score": float(scores[idx])
                    })
                return results
            except Exception:
                pass
                
        # 2. Sparse TF-IDF Search Fallback
        if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
            query_vec = self.tfidf_vectorizer.transform([query])
            sims = (self.tfidf_matrix * query_vec.T).toarray().flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(sims[idx])
                })
            return results
            
        return []
