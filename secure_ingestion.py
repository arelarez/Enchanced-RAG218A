import os
import re
import hashlib
import shutil
from datetime import datetime

DIR_STAGING = "./docs_staging"
DIR_INFECTED = "./docs_infected"
DIR_PROCESSED = "./docs_processed"

MALICIOUS_PATTERNS = [
    r"ignore previous instruction",
    r"system override",
    r"disregard context",
    r"you are now",
    r"forget everything",
    r"bypass security"
]

def setup_directories():
    for d in [DIR_STAGING, DIR_INFECTED, DIR_PROCESSED]:
        os.makedirs(d, exist_ok=True)

def scan_text_heuristics(text: str) -> bool:
    """Returns True if the text is SAFE, False if DANGEROUS."""
    text_lower = text.lower()
    for pattern in MALICIOUS_PATTERNS:
        if re.search(pattern, text_lower):
            return False # Poison detected!
    return True

def generate_file_hash(filepath: str) -> str:
    """Create digital fingerprints (hashes) of files for forensics."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_metadata_general(filename: str) -> dict:
    """Extracting basic metadata from documents."""
    return {
        "filename": filename,
        "ingest_time": datetime.now().isoformat(),
        "system_status": "verified"
    }

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Break long text into chunks with overlap to maintain context."""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

def secure_ingest_documents(chunk_size=400, overlap=50):
    setup_directories()
    files = [os.path.join(DIR_STAGING, f) for f in os.listdir(DIR_STAGING) if f.endswith(".txt")]
    
    if not files:
        print("Staging area is empty. No new documents.")
        return

    for filepath in files:
        filename = os.path.basename(filepath)
        file_hash = generate_file_hash(filepath)
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        if not scan_text_heuristics(text):
            print(f"[ALERT] Anomaly detected on {filename}. Moving to quarantine!")
            shutil.move(filepath, os.path.join(DIR_INFECTED, filename))
            continue

        base_metadata = extract_metadata_general(filename)
        chunks = chunk_text(text, chunk_size, overlap)
        
        safe_chunks = []
        safe_metadatas = []
        safe_ids = []

        for i, chunk in enumerate(chunks):
            if scan_text_heuristics(chunk):
                metadata = {
                    "source": filename,
                    "chunk_id": i,
                    "file_hash": file_hash,
                    "clearance_level": "cleared_layer_1",
                    **base_metadata
                }
                safe_chunks.append(chunk)
                safe_metadatas.append(metadata)
                safe_ids.append(f"{file_hash}_{i}")
            else:
                print(f"[ALERT] Chunk {i} on {filename} infected. Chunk is discarded.")

        if safe_chunks:
            collection.add(
                documents=safe_chunks,
                metadatas=safe_metadatas,
                ids=safe_ids
            )
            print(f"[SUCCESS] {len(safe_chunks)} chunk safe from {filename} has been ingested.")
            shutil.move(filepath, os.path.join(DIR_PROCESSED, filename))