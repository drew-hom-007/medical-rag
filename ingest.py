import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from rich.console import Console
from rich.panel import Panel

console = Console()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "medical_docs"

def chunk_text(text, source_name):
    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    chunks = []

    for sentence in sentences:
        if len(sentence) > 20:
            chunks.append({
                "text": sentence,
                "source": source_name
            })

    return chunks

def main():
    console.print(Panel("Starting ingestion pipeline", style="bold blue"))

    console.print("\n[bold]Step 1: Loading embedding model...[/bold]")
    model = SentenceTransformer(MODEL_NAME)
    console.print(f"Loaded model: {MODEL_NAME}")

    console.print("\n[bold]Step 2: Setting up ChromaDB...[/bold]")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)
    console.print("ChromaDB ready")

    console.print("\n[bold]Step 3: Loading and chunking documents...[/bold]")
    all_chunks = []
    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        source_name = file_path.name
        chunks = chunk_text(text, source_name)
        all_chunks.extend(chunks)
        console.print(f"  {source_name}: {len(chunks)} chunks")

    console.print(f"\nTotal chunks: {len(all_chunks)}")

    console.print("\n[bold]Step 4: Embedding and storing chunks...[/bold]")
    texts = [c["text"] for c in all_chunks]
    sources = [c["source"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    embeddings = model.encode(texts, show_progress_bar=True)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": s} for s in sources],
        ids=ids
    )

    console.print(Panel(f"Done! {len(all_chunks)} chunks stored in ChromaDB", style="bold green"))


if __name__ == "__main__":
    main()