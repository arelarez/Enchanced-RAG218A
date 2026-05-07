import os
import gc
import threading
import torch
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

app = FastAPI(title="Crypsix Enhanced RAG API", version="1.0")

SECRET_API_KEY = os.environ.get("MY_SECRET_API_KEY", "arel_secure_key_2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SECRET_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Invalid API Key!"
        )
    return api_key

class ChromaRAG:
    def __init__(self, persist_directory="./chroma_db", collection_name="rag_collection", embedding_model_name="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

        try:
             self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name, embedding_function=self.embedding_fn)

    def add_docs(self, ids: List[str], docs: List[str], metadatas: Optional[List[Dict[str,Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in docs]
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 3):
        if self.collection.count() == 0:
            return []

        res = self.collection.query(query_texts=[query_text], n_results=n_results)
        out = []

        if res['ids'] and len(res['ids']) > 0:
            for i in range(len(res['ids'][0])):
                out.append({
                    'id': res['ids'][0][i],
                    'document': res['documents'][0][i],
                    'metadata': res['metadatas'][0][i] if res['metadatas'] else {}
                })
        return out

print("Loading Cross-Encoder Re-ranker...")
CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-Multilingual-MiniLm-L12-v2', max_length=512)
RAG = ChromaRAG()

print("Loading the Main LLM Model...")
LLM_MODEL_PATH = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

class LLMWrapper:
    def __init__(self, model_path):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> dict:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            **kwargs
        )
        
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": full_text}

    def reload(self, base_model=None, adapter_dir=None):
        pass

MODEL = LLMWrapper(LLM_MODEL_PATH)

def format_prompt(instruction: str, user_input: str) -> str:
    return f"Instruction:\n{instruction}\n\nInput:\n{user_input}\n\nResponse:\n"

def make_rag_prompt(instruction: str, user_input: str, retrieved: List[Dict[str,Any]]) -> str:
    context = ""
    for i, r in enumerate(retrieved):
        context += f"[DOC {i+1} — id={r.get('id')}]: {r.get('document')} \n"

    prompt = (
        f"Use the following retrieved documents to answer the instruction. "
        f"If the documents do not contain relevant information, clearly say you don't know.\n\n"
        f"{context}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{user_input}\n\n"
        f"Response:\n"
    )
    return prompt

@app.get("/health")
async def health():
    return {"status": "ok", "loaded": True}

@app.post("/generate")
async def generate(req: dict, api_key: str = Depends(verify_api_key)):
    try:
        retrieved_final = []
        query_search = req.get('user_input', req.get('instruction', ''))
        max_new_tokens = req.get('max_new_tokens', 256)

        if req.get('use_rag', False):
            k_initial = 10
            initial_retrieved = RAG.query(query_search, n_results=k_initial)

            if initial_retrieved:
                input_pairs = [[query_search, doc['document']] for doc in initial_retrieved]
                
                with torch.inference_mode():
                    score_cross = CROSS_ENCODER_MODEL.predict(input_pairs)
                
                for i in range(len(initial_retrieved)):
                    initial_retrieved[i]['cross_score'] = float(score_cross[i])
                    
                initial_retrieved.sort(key=lambda x: x['cross_score'], reverse=True)
                
                k_final = int(req.get('rag_k', 3))
                retrieved_final = initial_retrieved[:k_final]
                
                del input_pairs, score_cross, initial_retrieved

        if retrieved_final:
            prompt = make_rag_prompt(req.get('instruction', ''), query_search, retrieved_final)
        else:
            prompt = format_prompt(req.get('instruction', ''), query_search)

        with torch.inference_mode():
            raw_out = MODEL.generate(prompt=prompt, max_new_tokens=max_new_tokens)
            
        generated_text = raw_out.get("generated_text", "")
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return {"generated_text": generated_text, "retrieved": retrieved_final}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/reload") 
async def reload_model(base_model: Optional[str] = None, adapter_dir: Optional[str] = None): 
    try:
        MODEL.reload(base_model=base_model, adapter_dir=adapter_dir) 
        return {"status": "reloaded"} 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")