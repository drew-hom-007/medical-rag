import os
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from retrieve import retrieve

console = Console()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

def ask(question):
    console.print(Panel(f"[bold]Question:[/bold] {question}", style="bold blue"))

    documents, metadatas = retrieve(question)

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

Question: {question}

Answer:"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    answer = response.json()["response"]

    console.print(Panel(answer, title="[bold green]Answer[/bold green]", style="green"))

    return answer


if __name__ == "__main__":
    while True:
        console.print("\n[bold]Enter your question (or type 'quit' to exit):[/bold]")
        question = input("> ").strip()

        if question.lower() == "quit":
            console.print("[bold]Goodbye![/bold]")
            break

        if question:
            ask(question)