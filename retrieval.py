# ============================================================
# retrieval.py
# RAG-Based University Question Answering System
# Father Moses Orshio Adasu University, Makurdi
# Author: Bem Thomas
# ============================================================
# PURPOSE OF THIS FILE:
# Builds and queries the knowledge base. It works in layers:
#
#   LAYER 1 - Stored knowledge base
#       (a) University PDFs placed in the docs/ folder
#       (b) Pages crawled from the university website and saved
#           in data/site_knowledge.json (built offline)
#       Both are chunked, embedded and searched semantically.
#
#   LAYER 2 - Academic web search (live, optional)
#       Trusted academic sources for course content questions.
#
#   LAYER 3 - University website and general web (live, optional)
#       Current announcements and news. Every live call has a
#       timeout and fails safely, so the stored knowledge base
#       can still answer when the live web is unreachable.
# ============================================================

import os
import re
import io
import json
from functools import lru_cache

# -- PyMuPDF ---------------------------------------------------
try:
    import fitz
except ImportError:
    fitz = None

# -- OCR (optional - only available with Tesseract) ------------
try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
    print("[INFO] OCR (Tesseract) is available for scanned PDFs.")
except ImportError:
    OCR_AVAILABLE = False
    print("[WARNING] pytesseract not installed. Scanned PDFs will be skipped.")

# -- FAISS (optional - falls back to numpy dot-product search) --
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False
    print("[WARNING] faiss not available - using numpy fallback for vector search.")

import numpy as np
import pickle

from tqdm import tqdm
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# ddgs is the maintained successor of duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# -- NLTK data (safe download) ---------------------------------
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
    STOP_WORDS = set()

    def word_tokenize(text):
        return text.lower().split()

# ============================================================
# CONFIGURATION
# ============================================================

DOCS_FOLDER = "docs"
SITE_KNOWLEDGE_PATH = os.path.join("data", "site_knowledge.json")
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class HFEmbedder:
    def __init__(self, model_name, hf_token=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ).astype("float32")
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        import numpy as np
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-12)


def _load_sbert():
    return HFEmbedder(EMBEDDING_MODEL_NAME)


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.4
WEB_TOP_K = 3

ACADEMIC_SOURCES = [
    "wikipedia.org", "khanacademy.org", "geeksforgeeks.org",
    "britannica.com", "coursera.org", "mit.edu", "stanford.edu",
    "investopedia.com", "sciencedirect.com", "researchgate.net"
]

UNIVERSITY_NAME = "Father Moses Orshio Adasu University Makurdi"
UNIVERSITY_NAME_OLD = "Benue State University Makurdi"

# Official pages fetched live on every question (Layer 3).
# Add specific pages that change often, for example a news page.
UNIVERSITY_URLS = [
    "https://www.moaum.edu.ng/",
]


# ============================================================
# STEP 1 - PDF LOADING AND TEXT EXTRACTION
# ============================================================

def load_pdfs_from_folder(folder_path):
    """
    Reads all PDF files from the docs/ folder page by page.
    Returns a list of dicts: {"text", "source", "page"}.
    """
    pages = []

    if fitz is None:
        print("[WARNING] PyMuPDF (fitz) is not installed. Cannot read PDFs.")
        return pages

    if not os.path.exists(folder_path):
        print(f"[WARNING] docs/ folder does not exist at '{folder_path}'.")
        return pages

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{folder_path}'.")
        return pages

    print(f"[INFO] Found {len(pdf_files)} PDF file(s). Extracting text...")

    for filename in tqdm(pdf_files, desc="Loading PDFs"):
        filepath = os.path.join(folder_path, filename)
        file_pages = 0
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                # Little or no text: probably a scanned page, try OCR
                if len(text) < 50:
                    if OCR_AVAILABLE:
                        try:
                            mat = fitz.Matrix(300 / 72, 300 / 72)
                            pix = page.get_pixmap(matrix=mat)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            text = pytesseract.image_to_string(img, lang='eng').strip()
                            if len(text) > 50:
                                print(f"[OCR] Extracted text from scanned page "
                                      f"{page_num + 1} of {filename}")
                        except Exception as ocr_err:
                            print(f"[OCR ERROR] Page {page_num+1} of "
                                  f"{filename}: {ocr_err}")
                            continue
                    else:
                        continue

                if len(text) < 50:
                    continue

                pages.append({"text": text, "source": filename, "page": page_num + 1})
                file_pages += 1
            doc.close()
        except Exception as e:
            print(f"[ERROR] Could not read {filename}: {e}")

        print(f"[INFO] {filename}: {file_pages} page(s) with usable text.")
        if file_pages == 0:
            print(f"[WARNING] {filename} gave no text. It may be a scanned PDF "
                  f"that needs OCR (install Tesseract + pytesseract).")

    print(f"[INFO] Extracted {len(pages)} pages total.")
    return pages


# ============================================================
# STEP 2 - TEXT CLEANING
# ============================================================

def clean_text(text):
    """Removes page numbers, odd characters and extra whitespace."""
    text = text.replace("\n", " ")
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+\d+\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================
# STEP 3 - CHUNKING
# ============================================================

def chunk_pages(pages):
    """Splits each page into overlapping chunks of CHUNK_SIZE characters."""
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
                chunks.append({"text": chunk_text, "source": source, "page": page_num})
            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    print(f"[INFO] Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


def load_site_chunks(path=SITE_KNOWLEDGE_PATH):
    """
    Loads the pre-crawled university website text (created by
    scrape_static.py) and splits it into the same chunk format used
    for PDFs. Returns an empty list if the file does not exist.
    """
    if not os.path.exists(path):
        print(f"[INFO] No site knowledge file at '{path}'. Skipping website content.")
        return []

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    out = []
    for rec in records:
        text = clean_text(rec.get("text", ""))
        source = rec.get("source", "website")
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) > 100:
                out.append({"text": piece, "source": source, "page": "site"})
            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    print(f"[INFO] Created {len(out)} chunks from {len(records)} website page(s).")
    return out


# ============================================================
# STEP 4 - EMBEDDING AND INDEX
# ============================================================

def build_faiss_index(chunks):
    """
    Embeds all chunks and saves the index. A numpy copy
    (faiss_index.bin.npy) is ALWAYS saved so the app can run on
    Streamlit Cloud without installing FAISS.
    """
    print("[INFO] Loading embedding model...")
    model = _load_sbert()

    print("[INFO] Generating embeddings for all chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = np.array(model.encode(texts, show_progress_bar=True), dtype="float32")
    dimension = embeddings.shape[1]

    # Always save the numpy version (used on the cloud)
    np.save(FAISS_INDEX_PATH + ".npy", embeddings)
    print(f"[INFO] Numpy index saved ({len(embeddings)} vectors).")

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"[INFO] FAISS index saved with {index.ntotal} vectors.")
    else:
        index = embeddings

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[INFO] Chunks saved to '{CHUNKS_PATH}'.")

    return index, chunks, model


# ============================================================
# STEP 5 - LOAD EXISTING INDEX (cached)
# ============================================================

@lru_cache(maxsize=1)
def load_index_and_chunks():
    """
    Loads the index, chunk list and embedder. The result is cached,
    so it is loaded once per server process instead of on every question.

    Order of attempts:
      1. FAISS index on disk (needs faiss)
      2. Numpy index on disk (works everywhere)
      3. Build from docs/ and data/site_knowledge.json (needs the
         embeddings API; slow, so commit the built index instead)
      4. Empty index (the live web layers can still answer)
    """
    npy_path = FAISS_INDEX_PATH + ".npy"

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH) and FAISS_AVAILABLE:
        print("[INFO] Loading FAISS index from disk...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"[INFO] Ready. Index contains {index.ntotal} vectors.")
        return index, chunks, _load_sbert()

    if os.path.exists(npy_path) and os.path.exists(CHUNKS_PATH):
        print("[INFO] Loading numpy index from disk...")
        index = np.load(npy_path)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"[INFO] Ready. Numpy index contains {len(index)} vectors.")
        return index, chunks, _load_sbert()

    has_docs = (os.path.exists(DOCS_FOLDER) and
                any(f.lower().endswith(".pdf") for f in os.listdir(DOCS_FOLDER)))
    has_site = os.path.exists(SITE_KNOWLEDGE_PATH)

    if has_docs or has_site:
        print("[INFO] No saved index found. Building knowledge base...")
        try:
            build_knowledge_base()
        except Exception as e:
            print(f"[WARNING] Could not build knowledge base: {e}")
        if os.path.exists(npy_path) and os.path.exists(CHUNKS_PATH):
            index = np.load(npy_path)
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks, _load_sbert()

    print("[WARNING] No index available. Starting with an empty index.")
    return np.zeros((0, EMBEDDING_DIM), dtype="float32"), [], _load_sbert()


# ============================================================
# STEP 6 - SEMANTIC RETRIEVAL (LAYER 1)
# ============================================================

def semantic_retrieve(query, index, chunks, model, top_k=TOP_K):
    """
    Embeds the query and returns the most similar chunks.
    Returns an empty list if the index is empty.
    """
    n_vecs = index.ntotal if FAISS_AVAILABLE and hasattr(index, "ntotal") else len(index)
    if n_vecs == 0:
        print("[INFO] Index is empty - no stored documents to search.")
        return []

    query_embedding = model.encode([query]).astype("float32")

    results = []
    if FAISS_AVAILABLE and hasattr(index, "search"):
        scores, indices = index.search(query_embedding, top_k)
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
    else:
        dot_scores = (index @ query_embedding.T).flatten()
        for idx in np.argsort(dot_scores)[::-1][:top_k]:
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "page": chunks[idx]["page"],
                "score": float(dot_scores[idx]),
                "retrieval_type": "semantic"
            })

    return results


# ============================================================
# STEP 7 - BM25 RETRIEVAL (baseline and fallback)
# ============================================================

def build_bm25_index(chunks):
    """Builds a BM25 keyword index from the same chunks."""
    tokenised = []
    for chunk in chunks:
        tokens = word_tokenize(chunk["text"].lower())
        tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
        tokenised.append(tokens)

    bm25 = BM25Okapi(tokenised)
    return bm25, tokenised


def bm25_retrieve(query, bm25, chunks, top_k=TOP_K):
    """Retrieves top_k chunks using BM25 keyword matching."""
    tokens = word_tokenize(query.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

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
# STEP 8 - LAYER 3: UNIVERSITY WEBSITE AND GENERAL WEB SEARCH
# ============================================================

def scrape_university_websites(query):
    """
    Fetches the pages listed in UNIVERSITY_URLS and returns passages
    that match the query. Fails safely: any error just means no
    live passages from that page.
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
                print(f"[WARNING] Could not access {url} - status {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

            if not text or len(text) < 100:
                continue

            chunk_size = 500
            for i in range(0, min(len(text), 5000), chunk_size):
                chunk = text[i:i + chunk_size].strip()
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
    """Runs a single web search. Returns [] on any error."""
    try:
        return list(DDGS().text(sq, max_results=top_k, region=region))
    except Exception as e:
        print(f"[WARNING] Web search failed for '{sq}': {e}")
        return []


def web_search_university(query, top_k=WEB_TOP_K):
    """LAYER 3 - live web search about the university."""
    results = []

    try:
        search_queries = [
            f"{UNIVERSITY_NAME} {query}",
            f"MOAUM Makurdi {query}",
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
                    search_results.extend(future.result())
                except Exception:
                    pass

        if not search_results:
            print("[INFO] No university results. Trying plain search...")
            search_results = _ddg_search_one(query, top_k, region="wt-wt")

        seen_urls = set()
        for result in search_results:
            snippet = result.get("body", "")
            url = result.get("href", "")
            title = result.get("title", "")
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
# STEP 9 - LAYER 2: ACADEMIC WEB SEARCH
# ============================================================

def web_search_academic(query, top_k=WEB_TOP_K):
    """LAYER 2 - live search of trusted academic sources."""
    results = []

    try:
        academic_query = f"{query} explanation definition university course"
        print(f"[INFO] Layer 2 academic search: '{academic_query}'")

        search_results = _ddg_search_one(academic_query, top_k * 2, region="wt-wt")

        trusted_results = []
        other_results = []

        for result in search_results:
            url = result.get("href", "")
            snippet = result.get("body", "")
            title = result.get("title", "")

            if len(snippet.strip()) < 50:
                continue

            entry = {
                "text": f"{title}. {snippet}",
                "source": url,
                "page": "web",
                "score": None,
                "retrieval_type": "web \u2014 academic"
            }

            if any(source in url for source in ACADEMIC_SOURCES):
                trusted_results.append(entry)
            else:
                other_results.append(entry)

        results = (trusted_results + other_results)[:top_k]

    except Exception as e:
        print(f"[WARNING] Academic web search failed: {e}")

    print(f"[INFO] Layer 2 returned {len(results)} result(s).")
    return results


# ============================================================
# STEP 10 - COMBINED RETRIEVAL PIPELINE
# ============================================================

def retrieve(query, index, chunks, model, use_web=True):
    """
    Combines all layers. If semantic retrieval fails (for example
    the embeddings API is unreachable), it falls back to BM25
    keyword retrieval over the same stored chunks.
    """
    print(f"\n[INFO] Starting retrieval for: '{query}'")

    print("[INFO] Layer 1: Searching stored knowledge base...")
    try:
        local_results = semantic_retrieve(query, index, chunks, model)
    except Exception as e:
        print(f"[WARNING] Semantic retrieval failed ({e}). Using BM25 fallback.")
        local_results = []
        if chunks:
            try:
                bm25, _ = build_bm25_index(chunks)
                local_results = bm25_retrieve(query, bm25, chunks)
            except Exception as e2:
                print(f"[WARNING] BM25 fallback failed: {e2}")
        for r in local_results:
            r["score"] = 0.0     # forces the web layers to run as well

    top_score = local_results[0]["score"] if local_results else 0.0
    print(f"[INFO] Layer 1 top score: {top_score:.4f}")

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
        search_results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_scrape = executor.submit(scrape_university_websites, query)
            future_layer3 = executor.submit(web_search_university, query, WEB_TOP_K)
            future_layer2 = (
                executor.submit(web_search_academic, query, WEB_TOP_K)
                if run_layer2 else None
            )

            print("[INFO] Layer 3: live university web sources (parallel)...")
            try:
                scraped_results = future_scrape.result(timeout=12)
            except Exception as e:
                print(f"[WARNING] Scrape failed: {e}")

            try:
                search_results = future_layer3.result(timeout=12)
            except Exception as e:
                print(f"[WARNING] Layer 3 search failed: {e}")

            if future_layer2:
                print("[INFO] Layer 2: academic web sources (parallel)...")
                try:
                    layer2_results = future_layer2.result(timeout=12)
                except Exception as e:
                    print(f"[WARNING] Layer 2 search failed: {e}")

        layer3_results = scraped_results + search_results

    combined = local_results + layer3_results + layer2_results

    print(f"[INFO] Total passages retrieved: {len(combined)}")
    print(f"  - Stored (Layer 1): {len(local_results)}")
    print(f"  - University web (Layer 3): {len(layer3_results)}")
    print(f"  - Academic web (Layer 2): {len(layer2_results)}")

    return combined


# ============================================================
# STEP 11 - KNOWLEDGE BASE BUILDER (run locally)
# ============================================================

def build_knowledge_base():
    """
    Offline pipeline:
        1. Load PDFs from docs/
        2. Load the crawled website text (data/site_knowledge.json)
        3. Chunk, embed and save chunks.pkl + faiss_index.bin.npy

    Run this on your own computer, then commit the output files.
    """
    print("=" * 60)
    print("Building Knowledge Base")
    print("=" * 60)

    pages = load_pdfs_from_folder(DOCS_FOLDER)
    chunks = chunk_pages(pages) + load_site_chunks()

    if not chunks:
        print("[ERROR] Nothing to index. Add PDFs to docs/ or run scrape_static.py.")
        return

    build_faiss_index(chunks)

    print("=" * 60)
    print("Knowledge base built successfully.")
    print(f"Total chunks indexed: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    build_knowledge_base()
