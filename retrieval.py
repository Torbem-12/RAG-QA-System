# ============================================================
# retrieval.py
# RAG-Based University Question Answering System
# Father Moses Orshio Adasu University, Makurdi
# Author: Bem Thomas
# ============================================================
# PURPOSE OF THIS FILE:
# This file handles everything related to building and querying
# the knowledge base. It operates on three layers:
#
#   LAYER 1 — Local University Documents (PDFs)
#       University handbooks, memos, circulars, lecture notes,
#       and handouts uploaded to the docs/ folder.
#
#   LAYER 2 — Academic Web Search
#       When a question is about course content, definitions,
#       or academic topics not fully covered in local PDFs,
#       the system searches trusted academic sources online
#       such as Wikipedia, Khan Academy, GeeksforGeeks,
#       and other university course pages.
#
#   LAYER 3 — University Website and General Web
#       Real-time search of the university's website and
#       social media handles for current announcements,
#       timetables, news, and administrative updates.
#
# Together these three layers ensure the system can answer
# both institutional questions AND academic content questions
# without requiring every document to be uploaded manually.
# ============================================================

import os
import re
import io

# ── PyMuPDF ──────────────────────────────────────────────────
try:
    import fitz
except ImportError:
    fitz = None

# ── OCR (optional — only available on Windows with Tesseract) ─
try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
    print("[INFO] OCR (Tesseract) is available for scanned PDFs.")
except ImportError:
    OCR_AVAILABLE = False
    print("[WARNING] pytesseract not installed. Scanned PDFs will be skipped.")

import faiss                         # vector index for semantic search
import numpy as np
import pickle                        # saves/loads the chunk list to disk

from tqdm import tqdm                # progress bar during indexing
from dotenv import load_dotenv       # loads API keys from .env file
from rank_bm25 import BM25Okapi      # keyword-based baseline retrieval
# SentenceTransformer is imported lazily inside load_index_and_chunks()
# to avoid loading PyTorch at module import time (kills Streamlit Cloud free tier)
from duckduckgo_search import DDGS   # free web search, no API key needed
from bs4 import BeautifulSoup        # parses HTML from web pages
import requests                      # fetches web page content
from concurrent.futures import ThreadPoolExecutor, as_completed  # parallel web search

# ── Load environment variables from .env ─────────────────────
load_dotenv()

# ── Download NLTK tokenizer data safely ──────────────────────
# We download to an explicit directory and wrap everything in
# try/except so a network failure on Streamlit Cloud does not
# crash the whole app at import time.
import nltk

_nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(_nltk_data_dir, exist_ok=True)

for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(_pkg, quiet=True, download_dir=_nltk_data_dir)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords as _sw
    from nltk.tokenize import word_tokenize
    STOP_WORDS = set(_sw.words("english"))
except Exception:
    # Fallback: empty stop-word set — BM25 still works, just noisier
    STOP_WORDS = set()
    def word_tokenize(text):
        return text.lower().split()

# ============================================================
# CONFIGURATION
# ============================================================

DOCS_FOLDER = "docs"
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

# Sentence-BERT model for generating semantic embeddings.
# all-MiniLM-L6-v2 was chosen because:
#   - Uses under 300 MB RAM — safe for a 4 GB laptop
#   - Runs on CPU without a GPU
#   - Produces 384-dimensional embeddings with strong
#     semantic quality for English text
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sbert():
    """Lazy-load SentenceTransformer so PyTorch is not imported at module level.
    This prevents Streamlit Cloud from crashing due to memory at startup."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# Maximum characters per chunk.
# 500 chosen because it fits within the language model's
# context window while being large enough to contain
# a complete thought or policy statement.
CHUNK_SIZE = 500

# Overlap between consecutive chunks in characters.
# 50-character overlap ensures sentences split across
# chunk boundaries are still fully represented.
CHUNK_OVERLAP = 50

# Number of top chunks retrieved from FAISS per query.
# 5 follows Lewis et al. (2020) who found this balances
# context richness against processing cost.
TOP_K = 5

# Similarity threshold below which web search is prioritised.
# 0.4 was chosen empirically — below this the retrieved
# chunk is likely only loosely related to the query.
CONFIDENCE_THRESHOLD = 0.4

# Number of web results to fetch per search call.
# Reduced from 5 to 3 per query for speed — 3 queries × 3 results
# still gives enough context while cutting network calls.
WEB_TOP_K = 3

# Trusted academic sources for course content searches.
# These sites are prioritised when a student asks about
# academic topics, definitions, or course concepts.
ACADEMIC_SOURCES = [
    "wikipedia.org",
    "khanacademy.org",
    "geeksforgeeks.org",
    "britannica.com",
    "coursera.org",
    "mit.edu",
    "stanford.edu",
    "investopedia.com",
    "sciencedirect.com",
    "researchgate.net"
]

# University-specific search targets for Layer 3
UNIVERSITY_NAME = "Father Moses Orshio Adasu University Makurdi"
UNIVERSITY_NAME_OLD = "Benue State University Makurdi"  # former name — more web presence

# Official university URLs to scrape directly
UNIVERSITY_URLS = [
    "https://www.bsum.edu.ng/",
    "https://www.bsum.edu.ng/news",
    "https://www.bsum.edu.ng/announcements",
    "https://www.bsum.edu.ng/about",
    "https://www.facebook.com/MOAUNewsonline",
]


# ============================================================
# STEP 1 — PDF LOADING AND TEXT EXTRACTION
# ============================================================

def load_pdfs_from_folder(folder_path):
    """
    Reads all PDF files from the docs/ folder and extracts
    their text content page by page using PyMuPDF.

    Returns a list of dicts, each with:
        - "text": raw text of one page
        - "source": filename of the PDF
        - "page": page number (1-indexed)
    """
    pages = []

    if fitz is None:
        print("[WARNING] PyMuPDF (fitz) is not installed. Cannot read PDFs.")
        return pages

    if not os.path.exists(folder_path):
        print(f"[WARNING] docs/ folder does not exist at '{folder_path}'.")
        return pages

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{folder_path}'.")
        return pages

    print(f"[INFO] Found {len(pdf_files)} PDF file(s). Extracting text...")

    for filename in tqdm(pdf_files, desc="Loading PDFs"):
        filepath = os.path.join(folder_path, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                # If page has little or no text, try OCR (scanned PDF)
                if len(text) < 50:
                    if OCR_AVAILABLE:
                        try:
                            # Render page as image at 300 DPI for good OCR quality
                            mat = fitz.Matrix(300 / 72, 300 / 72)
                            pix = page.get_pixmap(matrix=mat)
                            img_bytes = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_bytes))
                            text = pytesseract.image_to_string(img, lang='eng')
                            text = text.strip()
                            if len(text) > 50:
                                print(f"[OCR] Extracted text from scanned page "
                                      f"{page_num + 1} of {filename}")
                        except Exception as ocr_err:
                            print(f"[OCR ERROR] Page {page_num+1} of "
                                  f"{filename}: {ocr_err}")
                            continue
                    else:
                        continue  # Skip scanned page if OCR not available

                if len(text) < 50:
                    continue  # Still too little text after OCR, skip

                pages.append({
                    "text": text,
                    "source": filename,
                    "page": page_num + 1
                })
            doc.close()
        except Exception as e:
            print(f"[ERROR] Could not read {filename}: {e}")

    print(f"[INFO] Extracted {len(pages)} pages total.")
    return pages


# ============================================================
# STEP 2 — TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Removes noise from raw PDF text including page numbers,
    headers, special characters, and excessive whitespace.
    """
    text = text.replace("\n", " ")
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+\d+\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================
# STEP 3 — CHUNKING
# ============================================================

def chunk_pages(pages):
    """
    Splits each page into overlapping chunks of CHUNK_SIZE
    characters with CHUNK_OVERLAP characters of overlap.

    Character-based chunking is used instead of token-based
    because it is faster on low-resource hardware and
    produces approximately correct sizes (1 token ≈ 4 chars).
    """
    chunks = []
    for page in pages:
        text = clean_text(page["text"])
        source = page["source"]
        page_num = page["page"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 100:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page_num
                })
            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    print(f"[INFO] Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


# ============================================================
# STEP 4 — EMBEDDING AND FAISS INDEX
# ============================================================

def build_faiss_index(chunks):
    """
    Converts all chunks into SBERT embeddings and stores
    them in a FAISS IndexFlatIP index for exact inner
    product search.

    IndexFlatIP was chosen over approximate methods because:
    - For a few thousand university document chunks,
      exact search is fast enough (under 1 second)
    - Approximate methods are only needed for millions
      of vectors
    """
    print("[INFO] Loading SBERT embedding model...")
    model = _load_sbert()

    print("[INFO] Generating embeddings for all chunks...")
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # unit length for cosine similarity
    )

    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"[INFO] FAISS index built with {index.ntotal} vectors.")

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[INFO] FAISS index saved to '{FAISS_INDEX_PATH}'.")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[INFO] Chunks saved to '{CHUNKS_PATH}'.")

    return index, chunks, model


# ============================================================
# STEP 5 — LOAD EXISTING INDEX
# ============================================================

def load_index_and_chunks():
    """
    Loads the FAISS index, chunk list, and SBERT model
    from disk. Called once per question in rag_chain.ask().

    If the index files do not exist (e.g. on first run on
    Streamlit Cloud where they are not committed to GitHub),
    this function will:
      1. Try to build the index from any PDFs in the docs/ folder
      2. If there are no PDFs either, return an empty index so
         the app can still start and answer via web search

    This makes the app resilient to cold-start deployments
    where no pre-built index is present.
    """
    # ── Case 1: Index files exist — load them ────────────────
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("[INFO] Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)

        print("[INFO] Loading chunks from disk...")
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        print("[INFO] Loading SBERT model...")
        model = _load_sbert()

        print(f"[INFO] Ready. Index contains {index.ntotal} vectors.")
        return index, chunks, model

    # ── Case 2: No index — try building from docs/ folder ────
    docs_exist = (
        os.path.exists(DOCS_FOLDER) and
        any(f.endswith(".pdf") for f in os.listdir(DOCS_FOLDER))
        if os.path.exists(DOCS_FOLDER) else False
    )

    if docs_exist:
        print("[INFO] No saved index found. Building from docs/ folder...")
        build_knowledge_base()
        # Now load what we just built
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        model = _load_sbert()
        print(f"[INFO] Ready. Index contains {index.ntotal} vectors.")
        return index, chunks, model

    # ── Case 3: No index AND no docs — return empty index ────
    # The app can still answer questions using web search (Layers 2 & 3).
    # Layer 1 (local documents) will return no results until the user
    # uploads PDFs via the app's upload interface.
    print("[WARNING] No FAISS index and no PDF documents found.")
    print("[INFO] Starting with empty index — web search still works.")

    model = _load_sbert()
    dimension = 384  # all-MiniLM-L6-v2 embedding dimension
    index = faiss.IndexFlatIP(dimension)
    chunks = []

    return index, chunks, model


# ============================================================
# STEP 6 — SEMANTIC RETRIEVAL (LAYER 1)
# ============================================================

def semantic_retrieve(query, index, chunks, model, top_k=TOP_K):
    """
    Encodes the query using SBERT and searches the FAISS
    index for the most semantically similar document chunks.

    Returns chunks ranked by cosine similarity score.
    Returns an empty list if the index is empty (no documents uploaded yet).
    """
    # Guard: if the index is empty, skip FAISS search
    if index.ntotal == 0:
        print("[INFO] FAISS index is empty — no local documents to search.")
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(score),
            "retrieval_type": "semantic"
        })

    return results


# ============================================================
# STEP 7 — BM25 BASELINE RETRIEVAL
# ============================================================

def build_bm25_index(chunks):
    """
    Builds a BM25 keyword index from the same chunks used
    for FAISS. Used only for baseline evaluation comparison.
    """
    tokenised = []
    for chunk in chunks:
        tokens = word_tokenize(chunk["text"].lower())
        tokens = [t for t in tokens if t.isalpha()
                  and t not in STOP_WORDS]
        tokenised.append(tokens)

    bm25 = BM25Okapi(tokenised)
    return bm25, tokenised


def bm25_retrieve(query, bm25, chunks, top_k=TOP_K):
    """
    Retrieves top_k chunks using BM25 keyword matching.
    Used for baseline evaluation only.
    """
    tokens = word_tokenize(query.lower())
    tokens = [t for t in tokens if t.isalpha()
              and t not in STOP_WORDS]

    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(scores[idx]),
            "retrieval_type": "bm25"
        })

    return results


# ============================================================
# STEP 8 — LAYER 3: UNIVERSITY WEBSITE AND GENERAL WEB SEARCH
# ============================================================

def scrape_university_websites(query):
    """
    Directly visits the official university website and Facebook page,
    reads the full content, and returns passages relevant to the query.
    This gives more accurate and complete information than DuckDuckGo
    snippets which only show brief extracts.
    """
    results = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for url in UNIVERSITY_URLS:
        try:
            print(f"[INFO] Scraping: {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[WARNING] Could not access {url} — status {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script and style tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Extract all visible text
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

            if not text or len(text) < 100:
                continue

            # Split into chunks of ~500 characters
            chunk_size = 500
            for i in range(0, min(len(text), 5000), chunk_size):
                chunk = text[i:i+chunk_size].strip()
                if len(chunk) > 100:
                    query_words = query.lower().split()
                    chunk_lower = chunk.lower()
                    matches = sum(1 for w in query_words if w in chunk_lower)
                    if matches > 0 or i == 0:
                        results.append({
                            "text": chunk,
                            "source": url,
                            "page": "web",
                            "score": matches / max(len(query_words), 1),
                            "retrieval_type": "uni_web"
                        })

            print(f"[INFO] Scraped {len(results)} passages from {url}")

        except Exception as e:
            print(f"[WARNING] Could not scrape {url}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


def _ddg_search_one(sq, top_k, region="ng-en"):
    """
    Helper — runs a single DuckDuckGo text search.
    Designed to be called from a thread pool.
    Returns a list of raw result dicts, or [] on error.
    """
    try:
        ddgs_inst = DDGS()
        return list(ddgs_inst.text(sq, max_results=top_k, region=region))
    except Exception:
        return []


def web_search_university(query, top_k=WEB_TOP_K):
    """
    LAYER 3 — Searches the university website and general
    web for current institutional information including
    announcements, timetables, news, and admin updates.
    """
    results = []

    try:
        search_queries = [
            f"{UNIVERSITY_NAME} {query}",
            f"FMOAU Makurdi {query}",
            f"Benue State University Makurdi {query}",
        ]

        search_results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_ddg_search_one, sq, top_k): sq
                for sq in search_queries
            }
            for future in as_completed(futures, timeout=10):
                sq = futures[future]
                print(f"[INFO] Layer 3 search completed: '{sq}'")
                try:
                    r = future.result()
                    search_results.extend(r)
                except Exception:
                    pass

        if not search_results:
            print("[INFO] No university results. Trying plain search...")
            ddgs_inst = DDGS()
            search_results = list(ddgs_inst.text(query, max_results=top_k))

        seen_urls = set()
        for result in search_results:
            snippet = result.get("body", "")
            url     = result.get("href", "")
            title   = result.get("title", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if len(snippet.strip()) > 20:
                results.append({
                    "text": f"{title}. {snippet}",
                    "source": url,
                    "page": "web",
                    "score": None,
                    "retrieval_type": "uni_web"
                })

    except Exception as e:
        print(f"[WARNING] University web search failed: {e}")

    print(f"[INFO] Layer 3 returned {len(results)} result(s).")
    return results


# ============================================================
# STEP 9 — LAYER 2: ACADEMIC WEB SEARCH
# ============================================================

def web_search_academic(query, top_k=WEB_TOP_K):
    """
    LAYER 2 — Searches trusted academic sources online
    when a student asks about course content, definitions,
    concepts, or topics not covered in uploaded PDFs.
    """
    results = []

    try:
        academic_query = f"{query} explanation definition university course"
        print(f"[INFO] Layer 2 academic search: '{academic_query}'")

        ddgs_inst = DDGS()
        search_results = list(
            ddgs_inst.text(
                academic_query,
                max_results=top_k * 2,
            )
        )

        trusted_results = []
        other_results = []

        for result in search_results:
            url = result.get("href", "")
            snippet = result.get("body", "")
            title = result.get("title", "")

            if len(snippet.strip()) < 50:
                continue

            is_trusted = any(source in url for source in ACADEMIC_SOURCES)

            entry = {
                "text": f"{title}. {snippet}",
                "source": url,
                "page": "web",
                "score": None,
                "retrieval_type": "web — academic"
            }

            if is_trusted:
                trusted_results.append(entry)
            else:
                other_results.append(entry)

        combined = trusted_results + other_results
        results = combined[:top_k]

    except Exception as e:
        print(f"[WARNING] Academic web search failed: {e}")

    print(f"[INFO] Layer 2 returned {len(results)} result(s).")
    return results


# ============================================================
# STEP 10 — COMBINED THREE-LAYER RETRIEVAL PIPELINE
# ============================================================

def retrieve(query, index, chunks, model, use_web=True):
    """
    The main retrieval function — combines all three layers:

    LAYER 1: Semantic retrieval from local PDF knowledge base
    LAYER 2: Academic web search for course content
    LAYER 3: University website and general web search
    """
    print(f"\n[INFO] Starting three-layer retrieval for: '{query}'")

    print("[INFO] Layer 1: Searching local knowledge base...")
    local_results = semantic_retrieve(query, index, chunks, model)

    top_score = local_results[0]["score"] if local_results else 0.0
    print(f"[INFO] Layer 1 top similarity score: {top_score:.4f}")

    layer2_results = []
    layer3_results = []

    if use_web:
        academic_keywords = [
            "what is", "define", "definition", "explain",
            "types of", "difference between", "how does",
            "example of", "meaning of", "describe",
            "assignment", "course", "subject", "lecture",
            "concept", "theory", "formula", "calculate",
            "algorithm", "protocol", "method", "process"
        ]
        staff_keywords = [
            "vice chancellor", "vc ", "rector", "registrar",
            "dean", "provost", "bursar", "director", "head of",
            "professor", "dr.", "staff", "faculty", "who is",
            "who are", "contact", "office", "department"
        ]

        query_lower = query.lower()
        is_academic_query = any(kw in query_lower for kw in academic_keywords)
        is_staff_query = any(kw in query_lower for kw in staff_keywords)
        run_layer2 = top_score < CONFIDENCE_THRESHOLD or is_academic_query or is_staff_query

        scraped_results = []
        search_results  = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_scrape  = executor.submit(scrape_university_websites, query)
            future_layer3  = executor.submit(web_search_university, query, WEB_TOP_K)
            future_layer2  = (
                executor.submit(web_search_academic, query, WEB_TOP_K)
                if run_layer2 else None
            )

            print("[INFO] Layer 3: Scraping + searching university web sources (parallel)...")
            try:
                scraped_results = future_scrape.result(timeout=12)
            except Exception as e:
                print(f"[WARNING] Scrape failed: {e}")

            try:
                search_results = future_layer3.result(timeout=12)
            except Exception as e:
                print(f"[WARNING] Layer 3 search failed: {e}")

            if future_layer2:
                print("[INFO] Layer 2: Searching academic web sources (parallel)...")
                try:
                    layer2_results = future_layer2.result(timeout=12)
                except Exception as e:
                    print(f"[WARNING] Layer 2 search failed: {e}")

        layer3_results = scraped_results + search_results

    combined = local_results + layer3_results + layer2_results

    print(f"[INFO] Total passages retrieved: {len(combined)}")
    print(f"  - Local (Layer 1): {len(local_results)}")
    print(f"  - University web (Layer 3): {len(layer3_results)}")
    print(f"  - Academic web (Layer 2): {len(layer2_results)}")

    return combined


# ============================================================
# STEP 11 — KNOWLEDGE BASE BUILDER
# ============================================================

def build_knowledge_base():
    """
    Runs the full offline preprocessing pipeline:
        1. Load all PDFs from docs/ folder
        2. Clean and chunk the text
        3. Build and save the FAISS index

    Run this once, or whenever new documents are added.
    """
    print("=" * 60)
    print("Building Knowledge Base")
    print("=" * 60)

    pages = load_pdfs_from_folder(DOCS_FOLDER)

    if not pages:
        print("[ERROR] No pages loaded. Add PDFs to the docs/ folder.")
        return

    chunks = chunk_pages(pages)
    build_faiss_index(chunks)

    print("=" * 60)
    print("Knowledge base built successfully.")
    print(f"Total chunks indexed: {len(chunks)}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_knowledge_base()
