import os
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# TARGET_CHUNK_TOKENS defines the target size (in tokens) for each chunk of text that will be stored in the FAISS index.
#
# Why is this important?
# - Large language models (LLMs) have a fixed-size context window (e.g. 8k, 16k, or 128k tokens).
# - When using retrieval-augmented generation (RAG), we want to fill this window with chunks of knowledge from our documents.
# - If chunks are too small → too many short chunks waste space and cause noise.
# - If chunks are too large → we may exceed the context window, or LLM may miss important details inside a large chunk.
#
# Recommended values:
# - For models with ~8k context → 300-400 tokens per chunk.
# - For models with ~16k or larger → 400-700 tokens per chunk.
#
# The Retriever combines paragraphs until a chunk of approximately TARGET_CHUNK_TOKENS tokens is formed.
# The chunks are then embedded and stored in FAISS for similarity search.
#
# A value of 0 disables smart chunking:
#   - PDFs are split by paragraph (one per chunk)
#   - Text/Markdown files revert to a fixed default chunk size DEFAULT_CHUNK_TOKENS
TARGET_CHUNK_TOKENS = 300

# Default chunk size (in tokens) when smart chunking is disabled
DEFAULT_CHUNK_TOKENS = 300

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", data_dir="data", db_dir="db", window_size=0):
        self.model = SentenceTransformer(model_name)
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.index_path = self.db_dir / "index.faiss"
        self.meta_path = self.db_dir / "metadata.json"
        self.index = None
        self.metadata = []
        self.window_size = window_size

    def load_or_build(self, force_rebuild=False):
        self.db_dir.mkdir(exist_ok=True)

        if not force_rebuild and self.index_path.exists() and self.meta_path.exists():
            print("Loading existing index...")
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.meta_path.read_text())
        else:
            self._build_index()

    def _build_index(self):
        print("Building index...")
        print(f"Using TARGET_CHUNK_TOKENS = {TARGET_CHUNK_TOKENS} for all file types.")
        texts = []
        for file in self.data_dir.glob("*"):

            suffix = file.suffix.lower()

            if suffix == ".csv":
                print(f"Processing CSV: {file.name}")
                import csv
                with open(file, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        line = " → ".join(str(value).strip() for value in row.values() if value)
                        if line:
                            texts.append((file.name, f"{file.stem}: {line}"))
                continue

            elif suffix == ".pdf":
                print(f"Processing PDF: {file.name}")
                if TARGET_CHUNK_TOKENS > 0:
                    print(f"Using smart chunking (TARGET_CHUNK_TOKENS = {TARGET_CHUNK_TOKENS} tokens).")
                else:
                    print("Using paragraph-based chunking (TARGET_CHUNK_TOKENS = 0).")

                from pdfminer.high_level import extract_text
                full_text = extract_text(str(file))
                paragraphs = full_text.split("\n\n")

                if TARGET_CHUNK_TOKENS > 0:
                    current_chunk = ""
                    current_tokens = 0
                    import tiktoken
                    try:
                        enc = tiktoken.encoding_for_model("gpt-4")
                    except Exception:
                        enc = tiktoken.get_encoding("cl100k_base")

                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue

                        para_tokens = len(enc.encode(para))

                        if current_tokens + para_tokens > TARGET_CHUNK_TOKENS and current_chunk:
                            texts.append((file.name, current_chunk.strip()))
                            current_chunk = para
                            current_tokens = para_tokens
                        else:
                            current_chunk += " " + para
                            current_tokens += para_tokens

                    if current_chunk:
                        texts.append((file.name, current_chunk.strip()))
                else:
                    for para in paragraphs:
                        para = para.strip()
                        if para:
                            texts.append((file.name, para))
                continue

            elif suffix in [".md", ".txt"]:
                print(f"Processing file: {file.name}")
                chunks = self._split_text(file.read_text(), file.name)
                chunks = [(file.name, f"{file.stem}: {chunk}") for _, chunk in chunks]
                texts.extend(chunks)
                continue

            else:
                print(f"Skipping unsupported file type: {file.name}")
                continue

        if not texts:
            raise ValueError(
                "No text chunks found. Please make sure that .md or supported files are present in 'data/'.")

        embeddings = self.model.encode([t[1] for t in texts], show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata = [{"source": t[0], "text": t[1]} for t in texts]

        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2))
        print("Index saved.")

    def _split_text(self, text: str, source: str, chunk_size=TARGET_CHUNK_TOKENS if TARGET_CHUNK_TOKENS > 0 else DEFAULT_CHUNK_TOKENS) -> List[Tuple[str, str]]:
        # Simple chunking strategy based on TARGET_CHUNK_TOKENS (if > 0), otherwise fallback to paragraph-based for PDFs.
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append((source, chunk))
        return chunks

    def search(self, query: str, k=3) -> List[str]:
        if self.index is None:
            raise RuntimeError("Index is not loaded. Use load_or_build().")

        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)

        selected_indices = indices[0].tolist()

        if self.window_size > 0:
            expanded_indices = []
            for i in selected_indices:
                expanded_indices.extend(range(max(0, i - self.window_size), min(len(self.metadata), i + self.window_size + 1)))
            selected_indices = sorted(set(expanded_indices))

        return [self.metadata[i]["text"] for i in selected_indices if i != -1]