# ============================================================
# scrape_static.py
# Crawls the university website once (run on YOUR computer) and
# saves the text of each page to data/site_knowledge.json.
# retrieval.py then chunks and indexes that file.
#
# Usage:   python scrape_static.py
# ============================================================

import json
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

START = "https://www.moaum.edu.ng/"
MAX_PAGES = 40            # raise this if the site is bigger
HEADERS = {"User-Agent": "Mozilla/5.0"}
SKIP = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx",
        ".xls", ".xlsx", ".mp4", ".mp3")

DOMAIN = urlparse(START).netloc.replace("www.", "")


def same_site(url):
    return urlparse(url).netloc.replace("www.", "") == DOMAIN


seen, queue, records = set(), [START], []

while queue and len(records) < MAX_PAGES:
    url = queue.pop(0).split("#")[0]
    if url in seen or url.lower().endswith(SKIP):
        continue
    seen.add(url)

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("FAILED", url, repr(e))
        continue

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"]).split("#")[0]
        if same_site(link) and link not in seen:
            queue.append(link)

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    if len(text) > 200:
        records.append({"source": url, "text": text})
        print("OK", url, len(text), "characters")

    time.sleep(1)  # be polite to the server

os.makedirs("data", exist_ok=True)
with open("data/site_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(records)} page(s) to data/site_knowledge.json")
