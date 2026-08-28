from typing import List, Dict, Any, Optional
from src.genai.llm_client import LLMClient

SYSTEM_PROMPT = """
You are SynTwin AI Assistant, a domain-adaptive generative AI support agent connected to a Digital Twin decision system.
You are tasked with answering user questions by explaining the structured analytical context and retrieved company documents.

CRITICAL INSTRUCTIONS FOR TRUTH AND GROUNDING:
1. Grounding: Rely ONLY on the provided structured analytical context and retrieved company documents.
2. Hallucinations: Do NOT invent or hallucinate numbers, statistics, forecasts, predictions, SHAP values, or recommendations.
3. Insufficient Context: If the answer cannot be found in the provided contexts, respond strictly with: "I don't have enough information in the current SynTwin analysis to answer that reliably."
4. Missing RAG context: If a question is about company policy or organization terms but no relevant retrieved chunks are provided, respond strictly with: "I couldn't find supporting information in the uploaded company documents."
5. Fact vs Prediction: Clearly distinguish actual historical values from model predictions and future forecast projections.
6. Causality: Clearly distinguish statistical correlations (e.g. SHAP drivers, Pearson coefficients) from physical physical causation. Use wording like: "These variables strongly influence the model prediction." Do NOT use causative statements unless proven.
7. Concise Style: Provide concise, business-friendly, and professional explanations.
"""

CONVERSATIONAL_SYSTEM_PROMPT = """
You are SynTwin AI, a decision-intelligence assistant that helps analyze business data, explain model results, forecast trends, and evaluate decisions.
Provide a friendly, natural conversational response to the user's greeting or question.
Always identify yourself as SynTwin AI.
Keep your response short and welcoming.
"""

def is_conversational(query: str) -> bool:
    q = query.lower().strip().rstrip("?.!")
    # Basic greetings
    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "yo", "sup"}
    if q in greetings:
        return True
    
    # Conversational words
    conv_words = {"thanks", "thank you", "bye", "goodbye", "who are you", "your name", "whats your name", "what's your name"}
    if q in conv_words:
        return True
        
    # Checks for name introductions or name queries
    if "name" in q:
        return True
        
    # Other conversational markers
    identity_phrases = ["who are you", "what can you do", "what is your purpose", "how are you", "how's it going", "how are things", "nice to meet you", "thank you"]
    if any(p in q for p in identity_phrases):
        return True
        
    return False

def generate_grounded_response(llm_client: LLMClient, 
                              user_query: str, 
                              analytical_context_str: str, 
                              retrieved_chunks: List[Dict[str, Any]], 
                              chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Constructs the grounded prompt, invokes the LLM client, and compiles sources.
    
    Parameters:
    -----------
    llm_client : LLMClient
        Configured LLM client.
    user_query : str
        User's current question.
    analytical_context_str : str
        Aggregated Markdown context.
    retrieved_chunks : List[Dict]
        Matching text chunks retrieved via RAG.
    chat_history : List[Dict]
        Previous dialog logs.
        
    Returns:
    --------
    Dict[str, Any]
        Grounded response, matching sources list, and status flag.
    """
    chat_history = chat_history or []

    if is_conversational(user_query):
        # Conversational message: bypass grounding constraints and analytical context compilation
        history_str = ""
        if chat_history:
            history_str = "### Previous Conversation Messages:\n"
            for msg in chat_history[-5:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"
            history_str += "\n"
        prompt = f"{history_str}### User's Current Question:\n{user_query}\n"
        
        try:
            response_text = llm_client.generate(prompt, system_instruction=CONVERSATIONAL_SYSTEM_PROMPT)
            return {
                "response": response_text,
                "sources": [],
                "status": "success"
            }
        except Exception as e:
            return {
                "response": f"Failed to generate response: {str(e)}",
                "sources": [],
                "status": "error"
            }
    
    # 1. Compile Retrieved Chunks context
    rag_context = ""
    sources = []
    
    if retrieved_chunks:
        rag_context = "### Retrieved Company Documents Context Chunks:\n"
        for idx, chunk in enumerate(retrieved_chunks):
            # Only include chunks with positive relevance if using TF-IDF
            if chunk.get("score", 0.0) > 0.0 or llm_client.api_key is not None:
                rag_context += f"Document Source [{idx+1}]: {chunk['source']}\nContent: {chunk['text']}\n\n"
                sources.append(chunk["source"])
                
        sources = sorted(list(set(sources)))
        
    # 2. Build Chat History text
    history_str = ""
    if chat_history:
        history_str = "### Previous Conversation Messages:\n"
        for msg in chat_history[-5:]: # Keep last 5 messages for token thriftiness
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "\n"
        
    # 3. Assemble complete prompt
    prompt = f"""
{analytical_context_str}

{rag_context}

{history_str}
### User's Current Question:
{user_query}

Provide a grounded, professional response based ONLY on the guidelines above and context provided.
"""
    
    try:
        response_text = llm_client.generate(prompt, system_instruction=SYSTEM_PROMPT)
        return {
            "response": response_text,
            "sources": sources,
            "status": "success"
        }
    except Exception as e:
        return {
            "response": f"Failed to generate response: {str(e)}",
            "sources": [],
            "status": "error"
        }
