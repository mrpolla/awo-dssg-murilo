#!/usr/bin/env python3
"""
AWO website checker & Impressum/Kontakt scraper with multi-stage address detection.

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
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests
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
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/124.0"},
]

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE)
REQUEST_TIMEOUT = 15

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

def ensure_url(domain: str) -> Optional[str]:
    if not domain: return None
    d = domain.strip()
    if d.startswith("http://") or d.startswith("https://"): return d
    return f"https://{d}"

def get(url: str) -> Optional[requests.Response]:
    try:
        return requests.get(url, headers=pick_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None

def head(url: str) -> Optional[requests.Response]:
    try:
        return requests.head(url, headers=pick_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None

def site_reachable(url: str) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    r = head(url)
    if r is None or r.status_code >= 400 or r.status_code < 100:
        r = get(url)
    if r is None: return False, None, None, "no-response"
    if 200 <= r.status_code < 400: return True, r.url, r.status_code, r.reason
    return False, r.url, r.status_code, r.reason

def html_of(url: str) -> Optional[BeautifulSoup]:
    r = get(url)
    if r is None or not r.ok or not r.text: return None
    return BeautifulSoup(r.text, "lxml")

# --------------- Candidate pages ---------------

def find_candidate_pages(root_url: str, soup: Optional[BeautifulSoup]) -> Tuple[List[str], Optional[str], Optional[str]]:
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

    for path in COMMON_IMPRESSUM_PATHS + ["contact","kontakt-und-anfahrt","anfahrt","kontaktformular"]:
        cand = urljoin(root_url if root_url.endswith("/") else root_url+"/", path)
        r = head(cand)
        if r and 200 <= r.status_code < 400:
            full = r.url
            if full not in seen:
                seen.add(full); pages.append(full); tag(full)

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
        for i, ln in enumerate(base):
            m2 = PLZ_ORT_ANYWHERE.search(ln)
            if not m2: continue
            window = base[max(0, i-3):min(len(base), i+4)]
            best = None
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

# --------------- Core ---------------

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

def process_row(row: InputRow, polite_delay: float=1.0) -> ResultRow:
    input_url = ensure_url(row.domain) or ""
    reachable, final_url, code, reason = site_reachable(input_url) if input_url else (False, None, None, "no-domain")

    found_emails: List[str] = []
    best_addr: Dict[str,str] = {"street": None, "number": None, "plz": None, "ort": None, "_stage": None}
    impressum_url = None
    kontakt_url = None
    pages: List[str] = []

    if reachable and final_url:
        soup = html_of(final_url)
        if soup:
            pages, impressum_url, kontakt_url = find_candidate_pages(final_url, soup)
        else:
            pages = [final_url]

        all_cands: List[Dict[str,str]] = []
        print(f"  pages to visit: {pages}")
        for u in pages:
            print(f"    -> fetching: {u}")
            psoup = html_of(u)
            if not psoup: continue

            # emails
            emails_here = extract_emails(psoup)
            if emails_here: print(f"       emails: {emails_here}")
            found_emails = sorted(set(found_emails) | set(emails_here))

            # multi-stage addresses
            cands_here = extract_address_candidates_multistage(psoup, verbose=True)
            if cands_here:
                print(f"       address candidates: {cands_here}")
            all_cands.extend(cands_here)
            time.sleep(0.1)

        best_addr = pick_best_address(all_cands, row)
        print(f"  chosen best address (stage {best_addr.get('_stage')}): {{'street': {best_addr.get('street')}, 'number': {best_addr.get('number')}, 'plz': {best_addr.get('plz')}, 'ort': {best_addr.get('ort')}}}")
        if impressum_url: print(f"  impressum_url: {impressum_url}")
        if kontakt_url: print(f"  kontakt_url: {kontakt_url}")
        time.sleep(polite_delay)

    # choose best email for scoring (case-insensitive)
    best_email = ""
    if found_emails:
        dom = row.domain.split("/")[0].lower()
        candidates_em = [e for e in found_emails if any(x in e for x in [dom, "awo"])]
        best_email = candidates_em[0] if candidates_em else found_emails[0]
    print(f"  collected emails (deduped): {found_emails}")

    score_street = score_similarity(row.strasse, best_addr.get("street") or "")
    score_number = score_similarity(row.hausnummer, best_addr.get("number") or "")
    score_plz = score_similarity(row.plz, best_addr.get("plz") or "")
    score_ort = score_similarity(row.ort, best_addr.get("ort") or "")
    score_email = score_similarity(row.email.lower(), (best_email or "").lower())

    notes_parts = []
    if not reachable:
        notes_parts.append("unreachable")
    else:
        if score_plz < 60 or score_ort < 60: notes_parts.append("possible-address-change")
        if score_email < 60 and best_email: notes_parts.append("possible-email-change")
        if not found_emails: notes_parts.append("no-email-found")
        if not best_addr.get("street") or not best_addr.get("plz"): notes_parts.append("address-incomplete")

    return ResultRow(
        id=row.id,
        domain=row.domain,
        name=row.name,
        input_url=input_url,
        reachable=bool(reachable),
#        final_url=final_url,
        http_status=code,
        reason=reason,
        impressum_url=impressum_url,
        kontakt_url=kontakt_url,
#        pages_visited=";".join(pages),
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input/demo_seed.csv", help="Seed CSV (default: data/demo_seed.csv)")
    ap.add_argument("--output", default="data/output/scraped_results.csv", help="Results CSV")
    ap.add_argument("--max", type=int, default=0, help="Limit rows for a test run")
    ap.add_argument("--delay", type=float, default=1.0, help="Polite delay between sites (s)")
    args = ap.parse_args()

    rows: List[InputRow] = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        expected = {"id","domain","name","strasse","hausnummer","plz","ort","email"}
        missing = expected - set([c.strip().lower() for c in reader.fieldnames or []])
        if missing:
            print(f"[ERROR] CSV missing columns: {missing}", file=sys.stderr); sys.exit(2)
        for d in reader: rows.append(InputRow.from_dict(d))

    if args.max and args.max > 0: rows = rows[:args.max]

    results_pairs = []
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] Checking {r.domain} ...")
        try:
            res = process_row(r, polite_delay=args.delay)
            results_pairs.append((r, res))
        except Exception as e:
            print(f"  -> ERROR on {r.domain}: {e}", file=sys.stderr)
            res = ResultRow(
                    id=r.id,
                    domain=r.domain,
                    name=r.name,
                    input_url=ensure_url(r.domain) or "",
                    reachable=False,
                    http_status=None,
                    reason="exception",
                    found_street=None,
                    found_number=None,
                    found_plz=None,
                    found_ort=None,
                    impressum_url=None,
                    kontakt_url=None,
                    score_street=0,
                    score_number=0,
                    score_plz=0,
                    score_ort=0,
                    score_email=0,
                    notes="exception",
                    found_emails=""
                )
            results_pairs.append((r, res))
        time.sleep(0.2)

    out_fields = [
    "id","domain","name","strasse","hausnummer","plz","ort","email",
    "reachable","http_status","reason",
    "found_street","found_number","found_plz","found_ort","found_emails",
    "score_street","score_number","score_plz","score_ort","score_email",
    "impressum_url","kontakt_url","notes",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
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
    print(f"\nDone. Wrote {len(results_pairs)} rows to {args.output}")

if __name__ == "__main__":
    main()
