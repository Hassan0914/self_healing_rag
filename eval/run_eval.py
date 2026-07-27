"""
Evaluation harness for the Self-Healing RAG pipeline.

Runs a JSON set of {question, answerable} pairs against a live instance of
the API (must be running via `uvicorn app.main:app`) and reports:

  - Fallback rate on unanswerable questions (want: high, ideally 100%)
  - False-answer rate on unanswerable questions (want: 0% -- this is the
    hallucination metric that matters most)
  - Average combined faithfulness score on accepted answers
  - Average number of attempts per question (shows how often the retry loop fires)

Usage:
    python eval/run_eval.py --corpus_id my_corpus --eval_file eval/sample_eval_set.json
"""
import argparse
import json
import statistics
import sys

import requests


def run_eval(base_url: str, corpus_id: str, eval_file: str):
    with open(eval_file, "r") as f:
        data = json.load(f)

    questions = data["questions"]
    results = []

    for q in questions:
        resp = requests.post(
            f"{base_url}/ask",
            json={"question": q["question"], "corpus_id": corpus_id},
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"[ERROR] '{q['question']}' -> HTTP {resp.status_code}: {resp.text}")
            continue

        body = resp.json()
        results.append(
            {
                "question": q["question"],
                "answerable_ground_truth": q["answerable"],
                "was_fallback": body["was_fallback"],
                "final_answer": body["final_answer"],
                "total_attempts": body["total_attempts"],
                "final_combined_score": body["attempts"][-1]["critique"]["combined_score"],
            }
        )

    # --- Metrics ---
    unanswerable = [r for r in results if not r["answerable_ground_truth"]]
    answerable = [r for r in results if r["answerable_ground_truth"]]

    fallback_on_unanswerable = sum(r["was_fallback"] for r in unanswerable)
    false_answers_on_unanswerable = len(unanswerable) - fallback_on_unanswerable

    avg_attempts = statistics.mean(r["total_attempts"] for r in results) if results else 0
    avg_score_answerable = (
        statistics.mean(r["final_combined_score"] for r in answerable) if answerable else float("nan")
    )

    print("\n===== EVALUATION REPORT =====")
    print(f"Total questions evaluated: {len(results)}")
    print(f"Unanswerable questions:    {len(unanswerable)}")
    if unanswerable:
        print(
            f"  -> Correctly fell back:  {fallback_on_unanswerable}/{len(unanswerable)} "
            f"({100 * fallback_on_unanswerable / len(unanswerable):.1f}%)"
        )
        print(
            f"  -> Hallucinated instead: {false_answers_on_unanswerable}/{len(unanswerable)}  "
            "<- this is your hallucination rate on genuinely unanswerable Qs"
        )
    print(f"Answerable questions:      {len(answerable)}")
    print(f"  -> Avg combined faithfulness score: {avg_score_answerable:.3f}")
    print(f"Avg attempts per question (all):     {avg_attempts:.2f}")
    print("==============================\n")

    for r in results:
        flag = "FALLBACK" if r["was_fallback"] else "ANSWERED"
        print(f"[{flag}] ({r['total_attempts']} attempt(s)) {r['question']}")
        print(f"    -> {r['final_answer'][:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus_id", required=True)
    parser.add_argument("--eval_file", default="eval/sample_eval_set.json")
    args = parser.parse_args()

    try:
        run_eval(args.base_url, args.corpus_id, args.eval_file)
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {args.base_url}. Is the server running (uvicorn app.main:app)?")
        sys.exit(1)
