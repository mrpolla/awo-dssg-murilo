#!/usr/bin/env python3
"""
AWO website checker & Impressum/Kontakt scraper with multi-stage address detection.
Now supports resuming from existing results and incremental saving.

Run:
  python awo_scraper.py
…or:
  python awo_scraper.py --input your_seed.csv --output out.csv --delay 1.0
"""

import argparse
import csv
import random
import re
import sys
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import httpx
import asyncio
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from urllib.parse import urljoin

# ---------------- Config ----------------

COMMON_IMPRESSUM_PATHS = [
    "impressum","impressum/","impressum.html","imprint",
    "kontakt","kontakt/","kontakt.html",
    "kontakt-und-impressum","kontakt-impressum",
    "impressum-und-datenschutz","kontakt-und-datenschutz",
    "kontakt-impressum-datenschutz","wir/kontakt",
    "ueber-uns/impressum","verein/impressum","datenschutz-impressum",
]

HEADERS_ROTATING = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-de",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
]

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE)
REQUEST_TIMEOUT = 30
SAVE_INTERVAL = 10  # Save results every 10 entries
CONCURRENCY_LIMIT = 50  # how many sites at once



# Street tokens
STREET_TOKEN = r"(straße|strasse|str\.|weg|allee|platz|ufer|damm|gasse|ring|chaussee|steig|pfad|markt|berg)"
# Strict (whole line) match: "Hackenbroicher Str. 16"
STREET_LINE_STRICT = re.compile(
    rf"^(?P<street>[A-ZÄÖÜ][\wÄÖÜäöüß\.\- ]*{STREET_TOKEN})\s+(?P<number>\d+[a-zA-Z]?)\s*$"
, re.IGNORECASE)
# PLZ Ort (whole line) "50259 Pulheim"
PLZ_ORT_LINE = re.compile(r"^(?P<plz>\d{5})\s+(?P<ort>[A-ZÄÖÜ][\wÄÖÜäöüß\-\.\s]+)$", re.IGNORECASE)
# Single-line strict: "Hackenbroicher Str. 16, 50259 Pulheim"
SINGLE_LINE_STRICT = re.compile(
    rf"(?P<street>[A-ZÄÖÜ][\wÄÖÜäöüß\.\- ]*{STREET_TOKEN})\s+(?P<number>\d+[a-zA-Z]?)\s*[,|•|\- ]\s*(?P<plz>\d{{5}})\s+(?P<ort>[A-ZÄÖÜ][\wÄÖÜäöüß\-\.\s]+)"
, re.IGNORECASE)
# Relaxed (street+num appears anywhere in line; allow extra words around)
STREET_ANYWHERE_RELAX = re.compile(
    rf"(?P<street>[A-ZÄÖÜ][\wÄÖÜäöüß\.\- ]*{STREET_TOKEN})\s+(?P<number>\d+[a-zA-Z]?)"
, re.IGNORECASE)
# Relaxed PLZ Ort anywhere in line
PLZ_ORT_ANYWHERE = re.compile(r"(?P<plz>\d{5})\s+(?P<ort>[A-ZÄÖÜ][\wÄÖÜäöüß\-\.\s]+)", re.IGNORECASE)

# --------------- HTTP utils ---------------

def pick_headers() -> Dict[str, str]:
    return random.choice(HEADERS_ROTATING)


async def ensure_url(domain: str) -> Optional[str]:
    """
    Try both HTTPS and HTTP for incomplete URLs asynchronously.
    Returns the first working URL or defaults to HTTPS if both fail.
    """
    if not domain: return None
    
    d = domain.strip()
    
    if d.startswith("http://") or d.startswith("https://"): return d
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=REQUEST_TIMEOUT, verify=False) as client:
        for protocol, url in [("https", f"https://{d}"), ("http", f"http://{d}")]:
            try:
                r = await client.head(url, headers=pick_headers())
                if 200 <= r.status_code < 400: return url
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 405:
                    try:
                        async with client.stream("GET", url, headers=pick_headers()) as r:
                            if 200 <= r.status_code < 400: return url
                    except Exception:
                        continue
            except Exception:
                continue
    
    return f"https://{d}"


# async_client = httpx.AsyncClient(follow_redirects=True, timeout=REQUEST_TIMEOUT)
async def fetch(url: str):
    """Fetch page HTML asynchronously with concurrency control."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=REQUEST_TIMEOUT, verify=False) as client:
            r = await client.get(url, headers=pick_headers())
            if r.status_code < 400:
                return r
    except httpx.InvalidURL as e:
        print(f"Invalid URL skipped: {url} ({e})")
    except httpx.RequestError as e:
        print(f"Request failed: {url} ({e})")
    except Exception as e:
        # Catch-all for unexpected errors — optional but useful for debugging
        print(f"Unexpected error with {url}: {type(e).__name__} -> {e}")
    return None

async def html_of(url: str) -> Optional[BeautifulSoup]:
    r = await fetch(url)
    return BeautifulSoup(r.text, "lxml") if r else None

# --------------- Candidate pages ---------------

async def find_candidate_pages(root_url: str, soup: Optional[BeautifulSoup]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Return (pages_to_visit, impressum_url, kontakt_url)."""
    pages, seen = [root_url], {root_url}
    impressum_url, kontakt_url = None, None

    def tag(u: str, txt: str = ""):
        nonlocal impressum_url, kontakt_url
        lo = (u + " " + (txt or "")).lower()
        if "impressum" in lo and not impressum_url: impressum_url = u
        if ("kontakt" in lo or "contact" in lo) and not kontakt_url: kontakt_url = u

    if soup:
        for a in soup.find_all("a", href=True):
            href = a.get("href","")
            text = (a.get_text() or "").strip()
            if any(k in (href.lower()+" "+text.lower()) for k in ["impressum","kontakt","imprint","contact"]):
                full = urljoin(root_url, href)
                if full not in seen:
                    seen.add(full); pages.append(full); tag(full, text)

    tasks = [fetch(urljoin(root_url if root_url.endswith("/") else root_url+"/", path)) for path in COMMON_IMPRESSUM_PATHS + ["contact","kontakt-und-anfahrt","anfahrt","kontaktformular"]]
    responses = await asyncio.gather(*tasks)
    for r in responses:
        if r and 200 <= r.status_code < 400:
            full = str(r.url)
            if full not in seen:
                seen.add(full); pages.append(full); tag(full)

    # for path in COMMON_IMPRESSUM_PATHS + ["contact","kontakt-und-anfahrt","anfahrt","kontaktformular"]:
    #     cand = urljoin(root_url if root_url.endswith("/") else root_url+"/", path)
    #     r = await fetch(cand)
    #     if r and 200 <= r.status_code < 400:
    #         full = str(r.url)
    #         if full not in seen:
    #             seen.add(full); pages.append(full); tag(full)

    return pages, impressum_url, kontakt_url

# --------------- Extraction helpers ---------------

def extract_emails(soup: BeautifulSoup) -> List[str]:
    found = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            email = href.replace("mailto:", "").split("?")[0].strip().lower()
            if EMAIL_REGEX.fullmatch(email): found.add(email)
    text = soup.get_text(" ", strip=True)
    for m in EMAIL_REGEX.finditer(text):
        found.add(m.group(0).lower())
    return sorted(found)

def normalize(s: Optional[str]) -> str:
    return (s or "").strip()

def score_similarity(a: str, b: str) -> int:
    a = normalize(a).lower(); b = normalize(b).lower()
    if not a and not b: return 100
    if not a or not b: return 0
    return int(fuzz.token_sort_ratio(a, b))

def dedup_candidates(cands: List[Dict[str,str]]) -> List[Dict[str,str]]:
    seen, out = set(), []
    for c in cands:
        key = (c.get("street",""), c.get("number",""), c.get("plz",""), c.get("ort",""))
        if key not in seen and any(c.values()):
            out.append(c); seen.add(key)
    return out

# ---- Multi-stage address extraction ----
# Stage 1: strict line-by-line; Stage 2: single-line strict; Stage 3: relaxed inside-line;
# Stage 4: PLZ/Ort anywhere then look +/- 3 lines for street+number anywhere.

def extract_address_candidates_multistage(soup: BeautifulSoup, verbose: bool=False) -> List[Dict[str, str]]:
    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Filter only for phone/email noise (keep "Öffnungszeiten", etc.)
    def keep_line(ln: str) -> bool:
        low = ln.lower()
        return not any(x in low for x in ["tel:", "telefon", "fax", "e-mail", "email:", "mail:", "mailto:", "@ "])

    base = [ln for ln in lines if keep_line(ln)]

    candidates: List[Dict[str,str]] = []

    # --- Stage 1: strict line pairs (Street+Nr line, then PLZ Ort line within next 5)
    stage = 1
    for i, ln in enumerate(base):
        m1 = STREET_LINE_STRICT.match(ln)
        if m1:
            for j in range(i, min(i+6, len(base))):
                m2 = PLZ_ORT_LINE.match(base[j])
                if m2:
                    cand = {"street": m1.group("street").strip(),
                            "number": m1.group("number").strip(),
                            "plz": m2.group("plz").strip(),
                            "ort": m2.group("ort").strip(),
                            "_stage": stage}
                    candidates.append(cand); break
    if verbose and candidates:
        print(f"    [stage 1] {len(candidates)} address candidates")

    # --- Stage 2: strict single-line "Street Nr, PLZ Ort"
    if not candidates:
        stage = 2
        for ln in base:
            m = SINGLE_LINE_STRICT.search(ln)
            if m:
                candidates.append({
                    "street": m.group("street").strip(),
                    "number": m.group("number").strip(),
                    "plz": m.group("plz").strip(),
                    "ort": m.group("ort").strip(),
                    "_stage": stage
                })
        if verbose and candidates:
            print(f"    [stage 2] {len(candidates)} address candidates")

    # --- Stage 3: relaxed in-line: street+num anywhere + in same line PLZ Ort anywhere
    if not candidates:
        stage = 3
        for ln in base:
            m1 = STREET_ANYWHERE_RELAX.search(ln)
            m2 = PLZ_ORT_ANYWHERE.search(ln)
            if m1 and m2:
                candidates.append({
                    "street": m1.group("street").strip(),
                    "number": m1.group("number").strip(),
                    "plz": m2.group("plz").strip(),
                    "ort": m2.group("ort").strip(),
                    "_stage": stage
                })
        if verbose and candidates:
            print(f"    [stage 3] {len(candidates)} address candidates")

    # --- Stage 4: PLZ/Ort line anywhere, search nearby lines (+/-3) for street+num anywhere
    if not candidates:
        stage = 4
        plz_matches = [(i, PLZ_ORT_ANYWHERE.search(ln)) for i, ln in enumerate(base) if PLZ_ORT_ANYWHERE.search(ln)]
        for i, m2 in plz_matches:
            window = base[max(0, i-3):i+4]

            for ln2 in window:
                m1 = STREET_ANYWHERE_RELAX.search(ln2)
                if m1:
                    best = {
                        "street": m1.group("street").strip(),
                        "number": m1.group("number").strip(),
                        "plz": m2.group("plz").strip(),
                        "ort": m2.group("ort").strip(),
                        "_stage": stage
                    }
                    candidates.append(best)
                    break
        if verbose and candidates:
            print(f"    [stage 4] {len(candidates)} address candidates")

    # Dedup and return
    candidates = dedup_candidates(candidates)
    return candidates

# --------------- Data classes ---------------

@dataclass
class InputRow:
    id: str
    domain: str
    name: str
    strasse: str
    hausnummer: str
    plz: str
    ort: str
    email: str

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "InputRow":
        return InputRow(
            id=str(d.get("id","")).strip(),
            domain=str(d.get("domain","")).strip(),
            name=str(d.get("name","")).strip(),
            strasse=str(d.get("strasse","")).strip(),
            hausnummer=str(d.get("hausnummer","")).strip(),
            plz=str(d.get("plz","")).strip(),
            ort=str(d.get("ort","")).strip(),
            email=str(d.get("email","")).strip(),
        )

@dataclass
class ResultRow:
    id: str
    domain: str
    name: str
    input_url: str
    reachable: bool
    http_status: Optional[int]
    reason: Optional[str]
    found_street: Optional[str]
    found_number: Optional[str]
    found_plz: Optional[str]
    found_ort: Optional[str]
    impressum_url: Optional[str]
    kontakt_url: Optional[str]
    score_street: int
    score_number: int
    score_plz: int
    score_ort: int
    score_email: int
    notes: str
    found_emails: str

# --------------- Resume functionality ---------------

def load_existing_results(output_file: str) -> Set[str]:
    """Load existing results and return set of already processed IDs."""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_ids.add(row.get('id', '').strip())
            print(f"Found existing results file with {len(processed_ids)} processed entries")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
    return processed_ids

def save_results_incremental(results_pairs: List[Tuple['InputRow', 'ResultRow']], output_file: str, append: bool = False):
    """Save results to CSV file."""
    out_fields = [
        "id","domain","name","strasse","hausnummer","plz","ort","email",
        "reachable","http_status","reason",
        "found_street","found_number","found_plz","found_ort","found_emails",
        "score_street","score_number","score_plz","score_ort","score_email",
        "impressum_url","kontakt_url","notes",
    ]
    
    mode = 'a' if append else 'w'
    write_header = not (append and os.path.exists(output_file) and os.path.getsize(output_file) > 0)
    
    with open(output_file, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            w.writeheader()
        
        for inp, res in results_pairs:
            row_dict = {
                # original input fields first
                "id": inp.id,
                "domain": inp.domain,
                "name": inp.name,
                "strasse": inp.strasse,
                "hausnummer": inp.hausnummer,
                "plz": inp.plz,
                "ort": inp.ort,
                "email": inp.email,
                # results
                "reachable": res.reachable,
                "http_status": res.http_status,
                "reason": (res.reason or ""),
                "found_street": (res.found_street or ""),
                "found_number": (res.found_number or ""),
                "found_plz": (res.found_plz or ""),
                "found_ort": (res.found_ort or ""),
                "found_emails": (res.found_emails or ""),  # semicolon-separated
                "score_street": res.score_street,
                "score_number": res.score_number,
                "score_plz": res.score_plz,
                "score_ort": res.score_ort,
                "score_email": res.score_email,
                "impressum_url": (res.impressum_url or ""),
                "kontakt_url": (res.kontakt_url or ""),
                "notes": (res.notes or ""),
            }
            w.writerow(row_dict)

# --------------- Core ---------------

async def visit_page(page: str):
    psoup = await html_of(page)
    if not psoup:
        return [], []

    emails_here = extract_emails(psoup)
    cands_here = await asyncio.to_thread(extract_address_candidates_multistage, psoup)
    # cands_here = extract_address_candidates_multistage(psoup, verbose=True)
    return cands_here, emails_here

def pick_best_address(candidates: List[Dict[str,str]], seed: InputRow) -> Dict[str,str]:
    if not candidates:
        return {"street": None, "number": None, "plz": None, "ort": None, "_stage": None}
    def score(c):
        return (
            score_similarity(seed.strasse, c.get("street","")) +
            score_similarity(seed.hausnummer, c.get("number","")) +
            score_similarity(seed.plz, c.get("plz","")) +
            score_similarity(seed.ort, c.get("ort",""))
        )
    return max(candidates, key=score)

async def process_row(row: InputRow, polite_delay: float=5.0) -> ResultRow:
    input_url = await ensure_url(row.domain) or ""
    html_response = await fetch(input_url) if input_url else None
    reachable = bool(html_response)
    # reachable, final_url, code, reason = site_reachable(input_url) if input_url else (False, None, None, "no-domain")

    found_emails: List[str] = []
    best_addr: Dict[str,str] = {"street": None, "number": None, "plz": None, "ort": None, "_stage": None}
    impressum_url = None
    kontakt_url = None
    pages: List[str] = []
    score_street = ""
    score_number = ""
    score_plz = ""
    score_ort = ""
    score_email = ""

    if reachable and html_response:
        soup = BeautifulSoup(html_response.text, "lxml")
        pages, impressum_url, kontakt_url = await find_candidate_pages(input_url, soup)

        all_cands: List[Dict[str,str]] = []
        
        tasks = [visit_page(p) for p in pages]
        for task in asyncio.as_completed(tasks):
            cands_here, emails_here = await task
            all_cands.extend(cands_here)
            found_emails = sorted(set(found_emails) | set(emails_here))
            await asyncio.sleep(polite_delay) 

        best_addr = pick_best_address(all_cands, row)



    notes_parts = []
    if not reachable:
        notes_parts.append("unreachable")
    else:
        # choose best email for scoring (case-insensitive)
        best_email = ""
        if found_emails:
            dom = row.domain.split("/")[0].lower()
            candidates_em = [e for e in found_emails if any(x in e for x in [dom, "awo"])]
            best_email = candidates_em[0] if candidates_em else found_emails[0]
            # print(f"  collected emails (deduped): {found_emails}")

        score_street = score_similarity(row.strasse, best_addr.get("street") or "")
        score_number = score_similarity(row.hausnummer, best_addr.get("number") or "")
        score_plz = score_similarity(row.plz, best_addr.get("plz") or "")
        score_ort = score_similarity(row.ort, best_addr.get("ort") or "")
        score_email = score_similarity(row.email.lower(), (best_email or "").lower())

        if score_plz < 60 or score_ort < 60: notes_parts.append("possible-address-change")
        if score_email < 60 and best_email: notes_parts.append("possible-email-change")
        if not found_emails: notes_parts.append("no-email-found")
        if not best_addr.get("street") or not best_addr.get("plz"): notes_parts.append("address-incomplete")

    return row, ResultRow(
        id=row.id,
        domain=row.domain,
        name=row.name,
        input_url=input_url,
        reachable=bool(reachable),
        http_status=None,
        reason=None,
        impressum_url=impressum_url,
        kontakt_url=kontakt_url,
        found_emails=";".join(found_emails),
        found_street=best_addr.get("street"),
        found_number=best_addr.get("number"),
        found_plz=best_addr.get("plz"),
        found_ort=best_addr.get("ort"),
        score_street=score_street,
        score_number=score_number,
        score_plz=score_plz,
        score_ort=score_ort,
        score_email=score_email,
        notes=",".join(notes_parts),
    )



# --------------- Main ---------------

async def worker(item, semaphore, delay):
    async with semaphore:
        return await process_row(item, delay)

async def main(rows, batch_size, max_concurrency):
    semaphore = asyncio.Semaphore(max_concurrency)
    # Process rows with incremental saving
    batch_results = []
    total_processed = 0
    
    # Launch tasks
    tasks = [asyncio.create_task(worker(item, semaphore, args.delay)) for item in rows]
    
    for i, task in enumerate(asyncio.as_completed(tasks), start=1):
        # print(f"[{i}/{len(rows)}]")
        result = await task
        batch_results.append(result)

            
        # Save every SAVE_INTERVAL results or at the end
        if len(batch_results) >= batch_size or i == len(rows_to_process):

            print(f"\n--- Saving batch of {len(batch_results)} results to {args.output} ---")
            
            # Determine if we should append (when resuming) or overwrite
            append_mode = not (args.force and total_processed == len(batch_results))
            
            save_results_incremental(batch_results, args.output, append=append_mode)
            total_processed += len(batch_results)
            print(f"Saved {len(batch_results)} results (total processed in this session: {total_processed})")
            batch_results = []  # Clear the batch

            

    print(f"\nCompleted! Processed {total_processed} new entries in this session.")
    
    # Final summary
    if os.path.exists(args.output):
        total_in_file = sum(1 for _ in open(args.output, 'r', encoding='utf-8')) - 1  # -1 for header
        print(f"Total entries now in {args.output}: {total_in_file}")
        

if __name__ == "__main__":
    start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input/AWO_domains_with_impressum_data.csv", help="Seed CSV (default: data/demo_seed.csv)")
    ap.add_argument("--output", default="data/output/scraped_results.csv", help="Results CSV")
    ap.add_argument("--max", type=int, default=0, help="Limit rows for a test run")
    ap.add_argument("--delay", type=float, default=5.0, help="Polite delay between sites (s)")
    ap.add_argument("--force", action="store_true", help="Force restart from beginning (ignore existing results)")
    args = ap.parse_args()

    # Load input data
    rows: List[InputRow] = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        expected = {"id","domain","name","strasse","hausnummer","plz","ort","email"}
        missing = expected - set([c.strip().lower() for c in reader.fieldnames or []])
        if missing:
            print(f"[ERROR] CSV missing columns: {missing}", file=sys.stderr); sys.exit(2)
        for d in reader: rows.append(InputRow.from_dict(d))

    print(f"Loaded {len(rows)} total rows from input file")

    # Check for existing results and filter rows to process
    if not args.force:
        processed_ids = load_existing_results(args.output)
        rows_to_process = [r for r in rows if r.id not in processed_ids]
        print(f"Skipping {len(processed_ids)} already processed entries")
        print(f"Will process {len(rows_to_process)} remaining entries")
    else:
        rows_to_process = rows
        print("Force mode: processing all entries from scratch")

    if args.max and args.max > 0: 
        rows_to_process = rows_to_process[:args.max]
        print(f"Limited to {len(rows_to_process)} rows for testing")

    if not rows_to_process:
        print("No new rows to process. All entries are already completed!")
    else:

        asyncio.run(main(rows_to_process, batch_size=SAVE_INTERVAL, max_concurrency=CONCURRENCY_LIMIT))
    end = time.time()

print(f"Execution time: {end - start:.4f} seconds")