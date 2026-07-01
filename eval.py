from retrieve import retrieve
from rich.console import Console
from rich.table import Table

console = Console()

EVAL_SET = [
    {
        "question": "What is the first-line medication for type 2 diabetes?",
        "expected_keywords": ["metformin"],
        "expected_source": "diabetes.txt"
    },
    {
        "question": "What are the symptoms of asthma?",
        "expected_keywords": ["wheezing", "shortness of breath"],
        "expected_source": "asthma.txt"
    },
    {
        "question": "What score is used to assess pneumonia severity?",
        "expected_keywords": ["curb-65", "curb"],
        "expected_source": "pneumonia.txt"
    },
    {
        "question": "What medication is the cornerstone of heart failure treatment?",
        "expected_keywords": ["ace inhibitor", "ace inhibitors"],
        "expected_source": "heart_failure.txt"
    },
    {
        "question": "What blood pressure level defines hypertension?",
        "expected_keywords": ["130/80", "130"],
        "expected_source": "hypertension.txt"
    },
    {
        "question": "What inhaler is used for immediate asthma relief?",
        "expected_keywords": ["salbutamol"],
        "expected_source": "asthma.txt"
    },
    {
        "question": "What are the complications of uncontrolled diabetes?",
        "expected_keywords": ["kidney", "nerve", "stroke"],
        "expected_source": "diabetes.txt"
    },
    {
        "question": "What antibiotic treats community-acquired pneumonia?",
        "expected_keywords": ["amoxicillin"],
        "expected_source": "pneumonia.txt"
    },
    {
        "question": "What symptoms indicate heart failure?",
        "expected_keywords": ["breathlessness", "ankle swelling", "fatigue"],
        "expected_source": "heart_failure.txt"
    },
    {
        "question": "What lifestyle changes help treat hypertension?",
        "expected_keywords": ["sodium", "exercise", "alcohol"],
        "expected_source": "hypertension.txt"
    },
    {
        "question": "Which drug helps diabetic patients lose weight and protect their heart?",
        "expected_keywords": ["semaglutide", "glp-1"],
        "expected_source": "diabetes.txt"
    },
    {
        "question": "How do doctors measure average blood sugar over three months?",
        "expected_keywords": ["hba1c"],
        "expected_source": "diabetes.txt"
    },
    {
        "question": "What is a peak flow meter used for in asthma?",
        "expected_keywords": ["peak flow"],
        "expected_source": "asthma.txt"
    },
    {
        "question": "What CURB-65 score means a pneumonia patient needs hospital admission?",
        "expected_keywords": ["curb-65", "2"],
        "expected_source": "pneumonia.txt"
    },
    {
        "question": "What warning sign tells a heart failure patient they are retaining fluid?",
        "expected_keywords": ["weight", "2kg"],
        "expected_source": "heart_failure.txt"
    },
]


def evaluate():
    passed = 0
    failed = 0
    results = []

    for item in EVAL_SET:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        expected_source = item["expected_source"]

        documents, metadatas = retrieve(question)

        combined_text = " ".join(documents).lower()
        sources = [m["source"] for m in metadatas]

        keyword_hit = any(k in combined_text for k in expected_keywords)
        source_hit = expected_source in sources

        success = keyword_hit and source_hit

        if success:
            passed += 1
            status = "[bold green]PASS[/bold green]"
        else:
            failed += 1
            status = "[bold red]FAIL[/bold red]"

        results.append({
            "question": question[:60],
            "status": status,
            "keyword_hit": keyword_hit,
            "source_hit": source_hit,
            "top_source": sources[0] if sources else "none"
        })

    table = Table(title=f"Evaluation Results — {passed}/{len(EVAL_SET)} passed", show_lines=True)
    table.add_column("Question", style="white", width=40)
    table.add_column("Status", width=8)
    table.add_column("Keyword found", width=14)
    table.add_column("Right source", width=14)
    table.add_column("Top source retrieved", style="cyan", width=22)

    for r in results:
        table.add_row(
            r["question"],
            r["status"],
            "yes" if r["keyword_hit"] else "no",
            "yes" if r["source_hit"] else "no",
            r["top_source"]
        )

    console.print(table)
    score = round((passed / len(EVAL_SET)) * 100)
    console.print(f"\n[bold]Score: {score}%[/bold] ({passed} passed, {failed} failed)\n")


if __name__ == "__main__":
    evaluate()