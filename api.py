from fastapi import FastAPI
from pydantic import BaseModel
from retrieve import retrieve
import requests
from rich.console import Console

console = Console()

app = FastAPI(title="Medical RAG API")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str
    sources: list[str]

@app.get("/")
def health_check():
    return {"status": "Medical RAG API is running"}

@app.post("/ask")
def ask(question: Question):
    console.print(f"\n[bold]Received question:[/bold] {question.question}")

    documents, metadatas = retrieve(question.question)

    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context_parts.append(f"Source {i+1} ({meta['source']}):\n{doc}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful medical information assistant.
You will be given context from medical documents and a question.
Your job is to give a clear, direct answer using the information in the context.
If the context contains relevant information, use it confidently to answer.
If the context truly does not contain the answer, say so briefly in one sentence.
Always cite which source your answer comes from at the end.

Context:
{context}

Question: {question.question}

Answer:"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    answer_text = response.json()["response"]

    sources = list(set([m["source"] for m in metadatas]))

    console.print(f"[bold green]Answer generated[/bold green]")

    return Answer(
        question=question.question,
        answer=answer_text,
        sources=sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)