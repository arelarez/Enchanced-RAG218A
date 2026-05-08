import os
import threading
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEFAULT_ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs/tinyllama-qlora")
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_MAX_NEW_TOKENS = 256

def build_general_chat_prompt(tokenizer, user_input: str, retrieved_docs: list = None) -> str:
    """Build prompts that are compatible with RAG or regular Chat."""
    if retrieved_docs:
        context_str = "Reference Information:\n"
        for i, doc in enumerate(retrieved_docs):
            context_str += f"- {doc.get('document')}\n"
        
        system_instruction = (
            "You are a smart and helpful AI assistant. "
            "Use the Reference Information provided to answer the questions. "
            "If the answer is not in the reference, use your general knowledge but be honest."
        )
        final_user_content = f"{context_str}\n\n User Questions: {user_input}"
    else:
        system_instruction = "You are an intelligent, flexible AI assistant ready to help users."
        final_user_content = user_input

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": final_user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

class ModelWrapper: 
    def __init__(self, base_model=DEFAULT_BASE_MODEL, adapter_dir=DEFAULT_ADAPTER_DIR): 
        self.base_model = base_model 
        self.adapter_dir = adapter_dir 
        self._lock = threading.RLock() 
        self._loaded = False 
        self._tokenizer = None 
        self._pipeline = None 
        self._model = None

    def load(self, device_map=DEFAULT_DEVICE_MAP):
        with self._lock:
            if self._loaded:
                return
            try:
                compute_dtype = torch.bfloat16
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                )
            except Exception:
                base = AutoModelForCausalLM.from_pretrained(self.base_model, device_map=device_map)

            tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            try:
                model = PeftModel.from_pretrained(base, self.adapter_dir)
            except Exception:
                model = base

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map=device_map,
            )

            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True

    def reload(self, base_model=None, adapter_dir=None, device_map=DEFAULT_DEVICE_MAP):
        with self._lock:
            try:
                if self._model is not None:
                    del self._pipeline
                    del self._model
                    del self._tokenizer
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during memory cleanup: {e}")
                
            self._loaded = False
            if base_model:
                self.base_model = base_model
            if adapter_dir:
                self.adapter_dir = adapter_dir
            self.load(device_map=device_map)

    def generate(self, prompt: str, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True):
        if not self._loaded:
            self.load()

        with torch.inference_mode():
            out = self._pipeline(
                prompt,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                pad_token_id=self._tokenizer.eos_token_id,
            )

        return out[0]

MODEL = ModelWrapper()