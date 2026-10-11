# ============================================================
# rag_chain.py
# RAG-Based University Question Answering System
# Father Moses Orshio Adasu University, Makurdi
# Author: Bem Thomas
# ============================================================
# PURPOSE OF THIS FILE:
# This file connects the retrieval pipeline (retrieval.py)
# to the Groq API language model to form the complete
# Retrieval-Augmented Generation (RAG) chain.
#
# It has three jobs:
#   1. Take a student's question
#   2. Retrieve relevant passages from the knowledge base
#      and the web (via retrieval.py)
#   3. Send those passages + the question to the Groq API
#      (llama-3.1-8b-instant) and return a grounded answer
# ============================================================

import os
from groq import Groq                # Groq API client for LLM generation
from dotenv import load_dotenv       # loads API keys from .env file

# Import our retrieval functions from retrieval.py
from retrieval import (
    load_index_and_chunks,
    retrieve,
    build_bm25_index
)

# ── Load environment variables ────────────────────────────────
load_dotenv()

# ── Groq API Configuration ────────────────────────────────────

# The Groq API key is loaded from the .env file.
# Never hardcode API keys in source code — they must stay
# in the .env file which is never shared or uploaded.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# The language model used for response generation.
# llama-3.1-8b-instant was chosen because:
#   - It runs on Groq's inference infrastructure, not locally,
#     so it does not consume RAM on the student's laptop
#   - It delivers responses at up to 750 tokens/second,
#     making it faster than any locally run model
#   - It is an instruction-tuned model, meaning it follows
#     natural language instructions reliably
#   - It is freely available through Groq's free tier
GROQ_MODEL = "qwen/qwen3.8-27b"

# Maximum number of tokens the model can generate per response.
# 512 tokens is approximately 380 words — enough for a complete,
# detailed answer to most student questions without being
# so long that it becomes hard to read in the chat interface.
MAX_TOKENS = 512

# Temperature controls how creative/random the model's output is.
# 0.1 (very low) was chosen because:
#   - We want factual, consistent answers, not creative ones
#   - A low temperature makes the model stay close to the
#     information in the retrieved context
#   - Higher temperatures increase the risk of hallucination
TEMPERATURE = 0.1


# ============================================================
# TOPIC CLASSIFIER
# ============================================================

# This prompt instructs the language model to act as a
# topic classifier. It checks whether the student's question
# is within the allowed scope of the university assistant
# before any retrieval or answer generation takes place.
#
# Allowed scope: academics, university policies, fees, exams,
# courses, staff, registration, accommodation, student affairs,
# greetings, and general educational/subject questions that
# students commonly need help with.
#
# Out of scope: visa applications, immigration, unrelated
# government services, entertainment, politics, and any topic
# that has no connection to university life.
#
# Using the same language model (Groq) for classification
# means no extra library or tool is needed. The classifier
# call is lightweight — it only generates "yes" or "no".

CLASSIFIER_PROMPT = """You are a topic classifier for a university student assistant chatbot at Father Moses Orshio Adasu University (FMOAU), Makurdi, Nigeria.

Your job is to decide whether a student's question is within the allowed scope of this assistant.

ALLOWED topics (answer YES):
- University policies, rules, and regulations
- Fees, payments, and bursary matters
- Exam schedules, results, and academic calendar
- Courses, departments, and programmes
- Admission and registration
- Lecturers, staff, and university administration
- Student accommodation and welfare
- Academic subject questions (mathematics, science, programming, etc.)
- Study tips, assignments, and coursework help
- Greetings and general conversation
- Any topic that university students commonly need help with

NOT ALLOWED topics (answer NO):
- Visa applications and immigration processes
- Unrelated government services (passport, NIN, INEC, etc.)
- Entertainment, sports scores, music, movies
- Political opinions or news
- Medical diagnoses
- Any topic with no connection to university student life

Respond with ONLY the word YES or NO. Do not explain."""


# Keywords that immediately mark a question as OUT OF SCOPE
# without needing to call the Groq API at all.
# This fast pre-check saves one full network round-trip for
# clearly off-topic questions, improving response speed.
OUT_OF_SCOPE_KEYWORDS = [
    "visa", "passport", "immigration", "embassy", "consulate",
    "work permit", "travel document", "nin ", "national identity",
    "inec", "voter card", "pvc ", "election",
    "netflix", "spotify", "movie", "music chart", "celebrity",
    "bitcoin", "crypto", "forex", "stock market", "nse ",
    "betting", "sports bet", "premier league score",
    "recipe", "cook ", "cooking", "food recipe",
]

# Keywords that immediately mark a question as IN SCOPE
# without needing to call the Groq API.
# Greetings and clearly university-related words are caught here.
IN_SCOPE_KEYWORDS = [
    "university", "fmoau", "bsu ", "benue state university",
    "father adasu", "vice chancellor", "registrar", "department",
    "course", "exam", "fee", "tuition", "semester", "lecture",
    "assignment", "result", "cgpa", "matriculation", "admission",
    "hostel", "accommodation", "student", "faculty", "dean",
    "hello", "hi", "good morning", "good afternoon", "how are you",
    "what is", "define", "explain", "calculate", "algorithm",
    "programming", "python", "java", "mathematics", "physics",
    "chemistry", "biology", "engineering", "computer science",
]


def is_university_related(question):
    """
    Determines whether the student's question is within the
    allowed scope of the university assistant.

    OPTIMISATION — Two-stage check:

    Stage 1 (instant, no API call):
        Scans the question for known out-of-scope keywords
        (visa, passport, immigration, etc.) and known
        in-scope keywords (university, exam, course, etc.).
        If a match is found, returns immediately without any
        network call. This covers the majority of questions
        and costs essentially zero time.

    Stage 2 (Groq API call — only for ambiguous questions):
        If Stage 1 cannot decide, sends the question to the
        Groq API with the CLASSIFIER_PROMPT and uses the
        YES/NO response. This handles edge cases that keyword
        matching would get wrong.

    Parameters:
        question (str): The student's natural language question

    Returns:
        bool: True if the question is university-related,
              False if it is outside the allowed scope
    """
    q = question.lower().strip()

    # Stage 1a — fast out-of-scope check (no API needed)
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in q:
            print(f"[INFO] Topic classifier: OUT OF SCOPE (keyword match: '{kw}')")
            return False

    # Stage 1b — fast in-scope check (no API needed)
    for kw in IN_SCOPE_KEYWORDS:
        if kw in q:
            print(f"[INFO] Topic classifier: IN SCOPE (keyword match: '{kw}')")
            return True

    # Stage 2 — ambiguous question: ask the LLM
    print("[INFO] Topic classifier: ambiguous — calling Groq API...")
    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Question: {question}"}
            ],
            max_tokens=5,       # Only need YES or NO
            temperature=0.0     # Deterministic — always the same answer
        )

        result = response.choices[0].message.content.strip().upper()
        print(f"[INFO] Topic classifier API result: {result}")
        return result.startswith("Y")

    except Exception as e:
        # If the classifier fails for any reason (e.g. network error),
        # we allow the question through rather than blocking the user.
        print(f"[WARNING] Topic classifier error: {e}. Allowing question through.")
        return True


# ============================================================
# PROMPT TEMPLATE
# ============================================================

# This is the instruction given to the language model before
# every question. It tells the model:
#   1. What its role is (university information assistant)
#   2. What it must do (answer from context only)
#   3. What it must NOT do (invent answers, guess, hallucinate)
#   4. What to say when it does not know (be honest)
#
# This prompt design is critical for preventing hallucination.
# By explicitly instructing the model to use ONLY the provided
# context and to admit when it cannot find the answer,
# we ensure that students receive honest, traceable responses
# rather than confident but fabricated information.

SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable academic assistant for Father Moses Orshio Adasu University (FMOAU), Makurdi, Nigeria. This university was formerly known as Benue State University (BSU), Makurdi before being renamed. Web search results may refer to it by either name — treat both as the same institution.

KNOWN FACTS ABOUT FMOAU (use these when answering):
- Current Vice Chancellor: Professor Terseer Joseph Alabar
- Location: Makurdi, Benue State, Nigeria
- Official website: www.bsum.edu.ng
- The university is a Benue State Government institution
- Department of Mathematics and Computer Science is one of the departments

You will receive context passages from three sources:
1. Local university PDF documents
2. Web search results about the university
3. Academic web sources

RULES:

RULE 1 — GENERAL & CONVERSATIONAL QUESTIONS:
For greetings ("hi", "hello", "how are you"), general knowledge, subject explanations, maths, science, programming — answer directly from your own knowledge without referring to any documents.

RULE 2 — UNIVERSITY-SPECIFIC QUESTIONS:
For questions about FMOAU specifically (vice chancellor, staff, fees, exam dates, policies, announcements, departments, courses):
- READ ALL context passages carefully including web results
- If any context passage contains relevant information, USE it to answer
- If the web results contain the answer, state it clearly in your answer
- If NO context passage contains the answer, say: "I could not find this specific information in my current sources. Please visit the university website at fmoau.edu.ng or contact the relevant office directly."

RULE 3 — USE WEB RESULTS:
Web search results are real-time data. If a web result mentions a name, date, announcement or fact — USE it. Do not ignore web results.

RULE 4 — NEVER fabricate specific names, dates, or figures not found in context.

RULE 5 — FORMATTING (STRICTLY ENFORCED):
- Write in plain prose paragraphs ONLY
- Do NOT use asterisks (* or **) anywhere in your answer
- Do NOT use bullet points, dashes, or numbered lists
- Do NOT use markdown bold or italic formatting
- Do NOT write source references, URLs, or document names inside your answer
- Do NOT write "(Source 13)" or similar anywhere
- If you need to list multiple items, write them as a sentence: "The dates are X, Y, and Z."
- The system displays sources separately — never include them in your answer text

Always be warm, clear, and helpful."""


def build_prompt(question, retrieved_passages):
    """
    Constructs the full prompt sent to the language model.

    The prompt combines:
        1. The system instruction (SYSTEM_PROMPT above)
        2. The retrieved context passages with their sources
        3. The student's question

    This structure ensures the model always sees the context
    before the question, which helps it ground its answer
    in the retrieved material rather than its training memory.

    Parameters:
        question (str): The student's natural language question
        retrieved_passages (list): List of passage dicts from retrieve()

    Returns:
        str: The formatted context block to append to the user message
    """
    # Format each retrieved passage with its source label
    context_lines = []
    for i, passage in enumerate(retrieved_passages, 1):
        source = passage.get("source", "Unknown source")
        page = passage.get("page", "")
        retrieval_type = passage.get("retrieval_type", "")

        # Label sources cleanly — no full URLs in labels
        if retrieval_type in ("uni_web", "web", "academic", "web — university/general"):
            source_label = f"[Web Source {i}]"
        else:
            source_label = f"[Source {i}]"

        context_lines.append(f"{source_label}\n{passage['text']}")

    # Join all passages into one context block
    context_block = "\n\n".join(context_lines)

    # Build the user message combining context and question
    user_message = f"""Here are the relevant passages I found:

{context_block}

---

Based on the passages above, please answer this question:
{question}"""

    return user_message


def generate_answer(question, retrieved_passages):
    """
    Sends the question and retrieved passages to the Groq API
    and returns the language model's generated answer.

    This function implements the Generation stage of the
    RAG pipeline. The model receives the retrieved context
    and is instructed to generate an answer grounded in it.

    Parameters:
        question (str): The student's question
        retrieved_passages (list): Retrieved passages from retrieve()

    Returns:
        str: The generated answer from the language model
    """
    # Initialise the Groq client with the API key
    client = Groq(api_key=GROQ_API_KEY)

    # Build the formatted prompt
    user_message = build_prompt(question, retrieved_passages)

    # Send the request to the Groq API
    # The messages list follows the standard chat format:
    #   - "system": sets the model's behaviour and rules
    #   - "user": contains the context and the question
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    # Extract the generated text from the API response
    answer = response.choices[0].message.content
    return answer


def ask(question, use_web=True):
    """
    The main function called to answer a student's question.

    This function ties the entire RAG pipeline together:
        1. Checks if the question is within the allowed scope
           using the topic classifier (is_university_related)
        2. Loads the FAISS index and SBERT model
        3. Retrieves relevant passages (local + web)
        4. Generates an answer using the Groq API
        5. Returns the answer and the source passages

    The topic classifier runs BEFORE any retrieval takes place.
    If the question is out of scope, the system returns a polite
    refusal immediately without making any retrieval calls.
    This prevents the system from answering questions that have
    no connection to university life, such as visa applications.

    Parameters:
        question (str): The student's natural language question
        use_web (bool): Whether to include web augmentation
                        Default is True (always use web)

    Returns:
        dict: {
            "answer": the generated answer string,
            "sources": list of retrieved passage dicts
        }
    """
    print(f"\n[INFO] Question received: {question}")

    # ── Step 1: Topic Classification ─────────────────────────
    # Check if the question is within the allowed scope before
    # doing any retrieval or generation. This is the classifier
    # gate — out-of-scope questions are rejected here.
    print("[INFO] Running topic classifier...")
    if not is_university_related(question):
        print("[INFO] Question rejected by topic classifier — out of scope.")
        return {
            "answer": (
                "I am sorry, I can only assist with questions related to "
                "university matters such as academics, fees, exams, courses, "
                "policies, and student affairs at Father Moses Orshio Adasu "
                "University. Please ask a university-related question."
            ),
            "sources": []
        }

    # ── Step 2: Load the FAISS index, chunks, and SBERT model ─
    # These are loaded from disk (built by retrieval.py)
    index, chunks, model = load_index_and_chunks()

    # ── Step 3: Retrieve relevant passages ────────────────────
    # This combines semantic retrieval from FAISS and
    # real-time web augmentation via DuckDuckGo
    retrieved_passages = retrieve(
        query=question,
        index=index,
        chunks=chunks,
        model=model,
        use_web=use_web
    )

    if not retrieved_passages:
        return {
            "answer": "I could not find any relevant information. Please contact the university office directly.",
            "sources": []
        }

    print(f"[INFO] Retrieved {len(retrieved_passages)} passage(s) total.")

    # ── Step 4: Generate answer using Groq API ────────────────
    print("[INFO] Sending to Groq API for answer generation...")
    answer = generate_answer(question, retrieved_passages)

    print("[INFO] Answer generated successfully.")

    return {
        "answer": answer,
        "sources": retrieved_passages
    }


def ask_bm25(question):
    """
    Answers a question using BM25 keyword retrieval instead
    of semantic retrieval. Used for baseline evaluation only.

    This allows a direct comparison between:
        - Semantic RAG (FAISS + SBERT + Groq)
        - Keyword RAG  (BM25 + Groq)
    with all other components held constant.

    Parameters:
        question (str): The student's question

    Returns:
        dict: {"answer": str, "sources": list}
    """
    from retrieval import bm25_retrieve
    import pickle

    # Load chunks from disk
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Build BM25 index from chunks
    bm25, _ = build_bm25_index(chunks)

    # Retrieve using BM25 keyword matching
    retrieved_passages = bm25_retrieve(question, bm25, chunks)

    if not retrieved_passages:
        return {
            "answer": "I could not find any relevant information using keyword search.",
            "sources": []
        }

    # Generate answer using same Groq model
    answer = generate_answer(question, retrieved_passages)

    return {
        "answer": answer,
        "sources": retrieved_passages
    }


def ask_closedbook(question):
    """
    Answers a question using ONLY the language model's internal
    training knowledge, with NO retrieval at all.

    This is the closed-book baseline — it tests what happens
    when there is no retrieval augmentation. It is expected
    to produce hallucinated answers for university-specific
    questions because the model has no training data about
    Father Moses Orshio Adasu University specifically.

    This baseline directly demonstrates WHY retrieval
    augmentation is necessary for institutional QA systems.

    Parameters:
        question (str): The student's question

    Returns:
        dict: {"answer": str, "sources": []}
    """
    client = Groq(api_key=GROQ_API_KEY)

    # No context is provided — the model must answer from memory
    closed_book_prompt = f"""You are a university information assistant.
Answer this question about university policies and procedures to the best of your knowledge:

{question}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": closed_book_prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": []   # No sources — closed book has no retrieval
    }


# ============================================================
# ENTRY POINT — Test the RAG chain directly
# ============================================================

if __name__ == "__main__":
    # Test question — change this to any question about your
    # university documents to see how the system responds
    test_question = "What are the requirements for graduation?"

    print("=" * 60)
    print("Testing RAG Chain")
    print("=" * 60)

    result = ask(test_question)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print(f"SOURCES USED ({len(result['sources'])}):")
    print("=" * 60)
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. {source['source']} | Page: {source['page']} | Type: {source['retrieval_type']}")
