import os
import json
import re
import pypdf
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Normalizes whitespace and removes artifactual control characters."""
    if not text:
        return ""
    # Replace multiple spaces / non-breaking spaces
    text = text.replace('\xa0', ' ').replace('\u200b', ' ')
    # Normalize line breaks while keeping paragraph structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text_by_sentences(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> List[str]:
    """Splits text into chunks respecting sentence boundaries."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    
    # Split text by sentence terminators or newlines
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sent_len = len(sentence)
        if current_len + sent_len > max_chars and current_chunk:
            chunk_str = " ".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            
            # Form overlap from the tail of current_chunk
            overlap_list = []
            overlap_len = 0
            for prev_sent in reversed(current_chunk):
                if overlap_len + len(prev_sent) <= overlap_chars:
                    overlap_list.insert(0, prev_sent)
                    overlap_len += len(prev_sent)
                else:
                    break
            
            current_chunk = overlap_list + [sentence]
            current_len = sum(len(s) for s in current_chunk) + len(current_chunk)
        else:
            current_chunk.append(sentence)
            current_len += sent_len + 1
            
    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)
            
    return chunks

def build_corpus_chunks(inventory_path: str, dataset_base_dir: str, output_path: str) -> List[Dict[str, Any]]:
    with open(inventory_path, 'r', encoding='utf-8') as f:
        inventory = json.load(f)
        
    all_chunks = []
    total_pages = 0
    
    for doc in inventory:
        doc_path = os.path.join(dataset_base_dir, doc['relative_path'])
        if not os.path.exists(doc_path):
            print(f"Skipping missing file: {doc_path}")
            continue
            
        print(f"Processing: {doc['doc_slug']} ({doc['title'][:50]}...)")
        reader = pypdf.PdfReader(doc_path)
        doc_pages = len(reader.pages)
        total_pages += doc_pages
        
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            
            # Skip empty or negligible pages
            if len(cleaned) < 50:
                continue
                
            page_chunks = chunk_text_by_sentences(cleaned, max_chars=900, overlap_chars=120)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunk_id = f"{doc['doc_slug']}::p{page_num:02d}::c{chunk_idx+1:02d}"
                chunk_meta = {
                    "chunk_id": chunk_id,
                    "doc_slug": doc['doc_slug'],
                    "doc_title": doc['title'],
                    "authoring_body": doc['authoring_body'],
                    "year": doc['year'],
                    "category": doc['category'],
                    "subtopic": doc['subtopic'],
                    "page_number": page_num,
                    "chunk_index": chunk_idx + 1,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "text": chunk_text
                }
                all_chunks.append(chunk_meta)
                
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"\nCorpus processing complete:")
    print(f"- Total Documents: {len(inventory)}")
    print(f"- Total Pages: {total_pages}")
    print(f"- Total Deterministic Chunks: {len(all_chunks)}")
    print(f"- Average Chunk Size: {round(sum(c['char_count'] for c in all_chunks) / len(all_chunks), 1)} chars")
    return all_chunks

if __name__ == "__main__":
    inventory_file = 'd:/RAG/data/processed/document_inventory.json'
    base_dataset_dir = 'd:/RAG/ThyroGuard_Dataset'
    output_chunks_file = 'd:/RAG/data/processed/chunks.json'
    build_corpus_chunks(inventory_file, base_dataset_dir, output_chunks_file)
