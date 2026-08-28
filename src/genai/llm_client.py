import requests
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load local environment settings from the project root .env file
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

class LLMClient:
    """
    Abstractions for Generative AI LLM calls using direct HTTP requests.
    Supports Google Gemini and OpenAI.
    """
    def __init__(self, api_key: Optional[str] = None, 
                 provider: str = "gemini", 
                 model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("LLM_API_KEY") or st.secrets.get("GEMINI_API_KEY")
            except Exception:
                pass
                
        self.provider = provider.lower() if provider else "gemini"
        
        if self.provider == "gemini":
            self.model_name = model_name or os.environ.get("LLM_MODEL") or "gemini-3.6-flash"
        else: # openai
            self.model_name = model_name or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
            
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Invokes LLM text completion.
        """
        if not self.api_key:
            raise ValueError("LLM features require an API key. Please configure the LLM_API_KEY environment variable.")
            
        if self.provider == "gemini":
            return self._call_gemini(prompt, system_instruction)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_instruction)
        else:
            raise ValueError(f"Unsupported LLM provider '{self.provider}'. Supported providers are 'gemini', 'openai'.")
            
    def _call_gemini(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        # Use v1beta generateContent endpoint to support systemInstruction
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Gemini API returned error code {response.status_code}: {response.text}")
            
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise Exception(f"Failed to parse Gemini API response. Full response: {json.dumps(data)}")
            
    def _call_openai(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API returned error code {response.status_code}: {response.text}")
            
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception(f"Failed to parse OpenAI API response. Full response: {json.dumps(data)}")
