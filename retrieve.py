import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "medical_docs"
TOP_K = 5

def retrieve(query):
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K * 2,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    seen = set()
    unique_docs, unique_metas, unique_dists = [], [], []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_metas.append(meta)
            unique_dists.append(dist)
            if len(unique_docs) == TOP_K:
                break

    documents, metadatas, distances = unique_docs, unique_metas, unique_dists

    table = Table(title="Retrieved Chunks", show_lines=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Source", style="green", width=20)
    table.add_column("Similarity", style="magenta", width=12)
    table.add_column("Text", style="white")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = round((1 - dist) * 100, 1)
        table.add_row(
            str(i + 1),
            meta["source"],
            f"{similarity}%",
            doc[:200] + "..." if len(doc) > 200 else doc
        )

    console.print(table)
    return documents, metadatas


if __name__ == "__main__":
    retrieve("What medication is used to treat type 2 diabetes?")