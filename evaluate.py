import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunking import naive_chunk, structure_aware_chunk
from retrieval import Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(ROOT, "data", "diabetes_reference_document.md")
RESULTS = os.path.join(ROOT, "results")

QUESTIONS = [
    "What are the two main diagnostic criteria for Type 2 diabetes based on HbA1c and fasting plasma glucose?",
    "What HbA1c range indicates prediabetes?",
    "Name three first-line oral medications used to treat Type 2 diabetes.",
    "What are the classic symptoms of hyperglycemia?",
    "What lifestyle modifications are recommended for managing Type 2 diabetes?",
    "Name two microvascular complications of diabetes.",
    "Name two macrovascular complications of diabetes.",
    "How often should HbA1c be monitored in someone with stable Type 2 diabetes?",
    "What blood glucose target range is recommended for most adults with Type 2 diabetes?",
     "What symptoms should prompt someone to seek emergency medical care?",
]


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help="Generate answers using the OpenAI API (requires OPENAI_API_KEY).")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    with open(DOC_PATH, encoding="utf-8") as file:
        document = file.read()

    strategies = {
        "naive": naive_chunk(document, chunk_size=500, overlap=50),
        "structure_aware": structure_aware_chunk(document, max_size=1000),
    }

    if args.llm:
        from generation import answer_with_llm

    retrieval_rows, stats_rows = [], []

    for name, chunks in strategies.items():
        lengths = [len(chunk) for chunk in chunks]
        print(f"\n{'=' * 58}\n{name}: {len(chunks)} chunks")
        print(f"Average chunk size: {sum(lengths) / len(lengths):.2f} characters")

        stats_rows.append({
            "strategy": name,
            "chunk_count": len(chunks),
            "average_chunk_size_chars": round(sum(lengths) / len(lengths), 2),
            "min_chunk_size_chars": min(lengths),
            "max_chunk_size_chars": max(lengths),
            "questions_successful": "",
            "retrieval_accuracy_percent": "",
            "notes": "Fill accuracy after manually reviewing the retrieved evidence.",
        })

        retriever = Retriever(chunks)
        for qid, question in enumerate(QUESTIONS, start=1):
            matches = retriever.search(question, top_k=args.top_k)
            llm_answer = answer_with_llm(question, matches) if args.llm else ""
            print(f"\nQ{qid}: {question}")
            for match in matches:
                print(f"  Rank {match['rank']} | chunk {match['chunk_id']} | score {match['score']:.3f}")
                print("  " + " ".join(match["text"].split())[:180] + "...")

            for match in matches:
                retrieval_rows.append({
                    "strategy": name,
                    "question_id": qid,
                    "question": question,
                    "rank": match["rank"],
                    "chunk_id": match["chunk_id"],
                    "similarity_score": round(match["score"], 6),
                    "retrieved_text": match["text"],
                    "llm_answer": llm_answer,
                    "contains_correct_info": "",  # manually enter yes / partial / no
                    "answer_correct": "",  # manually grade if --llm was used
                })

    write_csv(os.path.join(RESULTS, "evaluation_results.csv"), retrieval_rows)
    write_csv(os.path.join(RESULTS, "chunk_statistics.csv"), stats_rows)
    print("\nSaved results/evaluation_results.csv")
    print("Saved results/chunk_statistics.csv")
    print("Manually label retrieved evidence and grade generated answers before reporting accuracy.")


if __name__ == "__main__":
    main()
