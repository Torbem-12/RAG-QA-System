# ============================================================
# evaluation.py — System Evaluation Script
# FMOAU RAG-Based Question Answering System
# Author : Bem Thomas | Supervisor: Dr. Stephen Akuma
# ============================================================

import time
import json
import statistics
from datetime import datetime
from rag_chain import ask, ask_bm25, ask_closedbook

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is Human-Computer Interaction?",
        "category": "Academic/Course Content",
        "expected_keywords": ["human", "computer", "interaction", "interface",
                              "design", "user"]
    },
    {
        "id": "Q2",
        "question": "What are the types of user interfaces in HCI?",
        "category": "Academic/Course Content",
        "expected_keywords": ["interface", "graphical", "command", "menu",
                              "touch", "voice"]
    },
    {
        "id": "Q3",
        "question": "What is usability in HCI?",
        "category": "Academic/Course Content",
        "expected_keywords": ["usability", "user", "effective", "efficient",
                              "satisf", "easy"]
    },
    {
        "id": "Q4",
        "question": "What is the difference between interface design and interaction design?",
        "category": "Academic/Course Content",
        "expected_keywords": ["design", "interface", "interaction", "user",
                              "visual", "behav"]
    },
    {
        "id": "Q5",
        "question": "What are the principles of good user interface design?",
        "category": "Academic/Course Content",
        "expected_keywords": ["principle", "consistency", "feedback",
                              "user", "simple", "design"]
    },
    {
        "id": "Q6",
        "question": "Who is the Vice Chancellor of the university?",
        "category": "University Information",
        "expected_keywords": ["vice", "chancellor", "professor",
                              "university", "alabar"]
    },
    {
        "id": "Q7",
        "question": "What is the name of the university?",
        "category": "University Information",
        "expected_keywords": ["father", "moses", "adasu", "university",
                              "makurdi", "benue"]
    },
    {
        "id": "Q8",
        "question": "What department offers Computer Science at the university?",
        "category": "University Information",
        "expected_keywords": ["mathematics", "computer", "science",
                              "department"]
    },
    {
        "id": "Q9",
        "question": "What is cognitive load in HCI?",
        "category": "Academic/Course Content",
        "expected_keywords": ["cognitive", "load", "mental", "memory",
                              "user", "information"]
    },
    {
        "id": "Q10",
        "question": "What are input and output devices used in HCI?",
        "category": "Academic/Course Content",
        "expected_keywords": ["input", "output", "keyboard", "mouse",
                              "screen", "device"]
    },
]


def compute_keyword_score(answer, keywords):
    if not answer or not keywords or not isinstance(answer, str):
        return 0.0
    answer_lower = answer.lower().strip()
    if not answer_lower:
        return 0.0
    matched = sum(1 for kw in keywords if str(kw).lower() in answer_lower)
    return round(matched / len(keywords), 3)


def compute_length_score(answer):
    if not answer or not isinstance(answer, str):
        return 0.0
    words = answer.strip().split()
    wc = len(words)
    if wc < 5:
        return 0.1
    elif wc < 20:
        return 0.5
    elif wc < 400:
        return 1.0
    else:
        return 0.9


def compute_refusal_score(answer):
    if not answer:
        return 0.0
    refusal_phrases = [
        "could not find", "cannot find", "not available",
        "no information", "do not have", "don't have",
        "please contact", "not in my", "cannot answer",
    ]
    answer_lower = answer.lower()
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return 0.0
    return 1.0


def compute_faithfulness(answer, sources):
    if not sources or not answer or not isinstance(answer, str):
        return 0.0

    source_text = " ".join([s.get("text", "") for s in sources[:3]])
    source_words = set(
        w.lower().strip(".,;:!?\"'()[]")
        for w in source_text.split() if len(w) >= 3
    )

    if not source_words:
        return 0.5

    answer_words = [
        w.lower().strip(".,;:!?\"'()[]")
        for w in answer.split() if len(w) >= 3
    ]
    if not answer_words:
        return 0.0

    overlap = sum(1 for w in answer_words if w in source_words)
    score = overlap / len(answer_words)
    return round(min(score * 3.5, 1.0), 3)


def compute_overall_score(keyword, length, refusal, faithfulness):
    return round(
        keyword     * 0.35 +
        faithfulness * 0.30 +
        length      * 0.20 +
        refusal     * 0.15,
        3
    )


def evaluate_system(system_name, ask_fn, questions):
    print("\n" + "=" * 60)
    print(f"Evaluating: {system_name}")
    print("=" * 60)

    results = []
    for q in questions:
        print(f"\n  [{q['id']}] {q['question'][:55]}...")

        start = time.time()
        try:
            result  = ask_fn(q["question"])
            answer  = result.get("answer", "") or ""
            sources = result.get("sources", []) or []
        except Exception as e:
            print(f"  ERROR: {e}")
            answer  = ""
            sources = []

        elapsed = round(time.time() - start, 2)

        keyword     = compute_keyword_score(answer, q["expected_keywords"])
        length      = compute_length_score(answer)
        refusal     = compute_refusal_score(answer)
        faithfulness = compute_faithfulness(answer, sources)
        overall     = compute_overall_score(keyword, length, refusal, faithfulness)

        print(f"  Time     : {elapsed}s")
        print(f"  Preview  : {answer[:80]}...")
        print(f"  Sources  : {len(sources)}")
        print(f"  Overall  : {overall}  |  Keyword: {keyword}  |  Faith: {faithfulness}  |  Length: {length}  |  Refusal: {refusal}")

        results.append({
            "question_id":    q["id"],
            "question":       q["question"],
            "category":       q["category"],
            "system":         system_name,
            "answer":         answer[:500],
            "num_sources":    len(sources),
            "response_time":  elapsed,
            "keyword_score":  keyword,
            "length_score":   length,
            "refusal_score":  refusal,
            "faithfulness":   faithfulness,
            "overall_score":  overall,
        })

    return results


def summarise(results, system_name):
    rows = [r for r in results if r["system"] == system_name]
    if not rows:
        return {"system": system_name}

    def avg(key):
        return round(statistics.mean(r[key] for r in rows), 3)

    return {
        "system":             system_name,
        "avg_overall":        avg("overall_score"),
        "avg_faithfulness":   avg("faithfulness"),
        "avg_keyword":        avg("keyword_score"),
        "avg_length":         avg("length_score"),
        "avg_refusal":        avg("refusal_score"),
        "avg_response_time":  avg("response_time"),
        "total_questions":    len(rows),
    }


def print_table(summaries):
    smap = {s["system"]: s for s in summaries}
    rag  = smap.get("RAG System (Semantic + Web)", {})
    bm25 = smap.get("BM25 Keyword Baseline", {})
    cb   = smap.get("Closed-Book Baseline", {})

    print("\n\n" + "=" * 72)
    print("EVALUATION RESULTS — COMPARISON TABLE")
    print("=" * 72)
    print(f"{'Metric':<26} {'RAG System':>13} {'BM25 Baseline':>14} {'Closed-Book':>13}")
    print("-" * 72)

    rows = [
        ("Overall Score",        "avg_overall"),
        ("Faithfulness",         "avg_faithfulness"),
        ("Keyword Relevance",    "avg_keyword"),
        ("Answer Completeness",  "avg_length"),
        ("Non-Refusal Rate",     "avg_refusal"),
        ("Avg Response Time(s)", "avg_response_time"),
    ]
    for label, key in rows:
        rv = rag.get(key,  0.0)
        bv = bm25.get(key, 0.0)
        cv = cb.get(key,   0.0)
        print(f"{label:<26} {rv:>13.3f} {bv:>14.3f} {cv:>13.3f}")

    print("=" * 72)
    print(f"\nDate: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    print(f"Questions: {rag.get('total_questions', 0)}")
    print(f"Model: qwen/qwen3.8-27b (Groq API)")
    print(f"Embedding: sentence-transformers/all-MiniLM-L6-v2")


def main():
    print("\n" + "=" * 60)
    print("FMOAU RAG SYSTEM EVALUATION")
    print(f"Started: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    print("=" * 60)

    all_results = []

    # Run all three systems
    all_results.extend(evaluate_system(
        "RAG System (Semantic + Web)",
        lambda q: ask(q, use_web=True),
        TEST_QUESTIONS
    ))
    all_results.extend(evaluate_system(
        "BM25 Keyword Baseline",
        ask_bm25,
        TEST_QUESTIONS
    ))
    all_results.extend(evaluate_system(
        "Closed-Book Baseline",
        ask_closedbook,
        TEST_QUESTIONS
    ))

    summaries = [
        summarise(all_results, "RAG System (Semantic + Web)"),
        summarise(all_results, "BM25 Keyword Baseline"),
        summarise(all_results, "Closed-Book Baseline"),
    ]

    print_table(summaries)

    output = {
        "evaluation_date":   datetime.now().isoformat(),
        "num_questions":     len(TEST_QUESTIONS),
        "summaries":         summaries,
        "detailed_results":  all_results,
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: evaluation_results.json")


if __name__ == "__main__":
    main()
