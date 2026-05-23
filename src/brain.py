# src/brain.py
import logging
import numpy as np
import pandas as pd
import streamlit as st
from google.genai.errors import APIError
from src.config import get_gemini_client

logger = logging.getLogger("FoodMetriX_Core.Brain")
client = get_gemini_client()

def compute_and_store_embeddings(df_or_chunks, filename, tenant_name, is_tabular=True):
    """Generates vectors via text-embedding-004 and saves them to session state."""
    if tenant_name not in st.session_state.vector_store:
        st.session_state.vector_store[tenant_name] = []
        
    count = 0
    if is_tabular:
        columns = df_or_chunks.columns.tolist()
        for idx, row in df_or_chunks.iterrows():
            kv_pairs = [f"'{col}' is '{row[col]}'" for col in columns if pd.notna(row[col])]
            serialized = f"In factory file {filename} (Row {idx+1}): {', '.join(kv_pairs)}."
            _embed_and_cache(serialized, filename, tenant_name)
            count += 1
    else:
        for chunk in df_or_chunks:
            serialized = f"In factory file {filename} ({chunk['meta']}): {chunk['text']}"
            _embed_and_cache(serialized, filename, tenant_name)
            count += 1
            
    return count

def _embed_and_cache(text, filename, tenant_name):
    try:
        resp = client.models.embed_content(model="text-embedding-004", contents=text)
        st.session_state.vector_store[tenant_name].append({
            "vector": resp.embeddings[0].values,
            "text": text,
            "filename": filename
        })
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")

def search_similarity(user_query, tenant_name, top_k=4):
    """Finds mathematically similar historical documentation blocks."""
    if tenant_name not in st.session_state.vector_store or not st.session_state.vector_store[tenant_name]:
        return "No local context data found."
        
    query_resp = client.models.embed_content(model="text-embedding-004", contents=user_query)
    query_vec = np.array(query_resp.embeddings[0].values)
    
    scored = []
    for item in st.session_state.vector_store[tenant_name]:
        item_vec = np.array(item["vector"])
        sim = np.dot(query_vec, item_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(item_vec))
        scored.append((sim, item["text"]))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([chunk[1] for chunk in scored[:top_k]])