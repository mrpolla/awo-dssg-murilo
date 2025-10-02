#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AWO Entity Deduplication - Creates a master list of unique entities
Each row represents one unique entity with data from all sources where it was found

Output columns (exact order):
EntityID, IsFacility, IsAssociation, IsLegalEntity, IsAWODomain, EntityName,
F_names, A_names, L_names, D_names,
[all F_* columns], [all A_* columns], [all L_* columns], [all D_* columns],
Members, N_members
"""

import os
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ------------- Config -------------
# Conservative fuzzy thresholds
FUZZY_NAME_MIN = 92
FUZZY_STREET_MIN = 85

ROOT = Path.cwd()

IN_FAC_ASS = os.path.join(
    ROOT, "data/input/2025_09_16_Einrichtunsdatenbank_Export_descriptions_final.xlsx"
)
IN_LEGAL = os.path.join(
    ROOT, "data/input/2026_09_18_JurPers_Fragebogen2025_descriptions_final.xlsx"
)
IN_DOMAINS = os.path.join(
    ROOT, "data/input/AWO_domains_with_impressum_data.xlsx"
)

OUT_DIR = os.path.join(ROOT, "data/output")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_XLSX = os.path.join(OUT_DIR, "deduplicated_entities.xlsx")
OUT_TXT_EXACT = os.path.join(OUT_DIR, "exact_name_matches.txt")
OUT_TXT_ADDR = os.path.join(OUT_DIR, "address_matches.txt")
OUT_TXT_FUZZY = os.path.join(OUT_DIR, "fuzzy_name_matches.txt")

MAX_PER_SOURCE = None  # e.g., take at most 300 rows from each source during testing


def _limit(df: pd.DataFrame, max_n) -> pd.DataFrame:
    if not max_n:  # handles None, 0, False
        return df
    n = min(int(max_n), len(df))
    return df.sample(n=n, random_state=42)

# ------------- Fuzzy similarity -------------
try:
    from rapidfuzz import fuzz
    def sim(a: str, b: str) -> int:
        return int(fuzz.token_set_ratio(a, b))
except Exception:
    import difflib
    def sim(a: str, b: str) -> int:
        return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)

# ------------- Normalization helpers -------------
UML = {"ä":"ae","ö":"oe","ü":"ue","ß":"ss","Ä":"Ae","Ö":"Oe","Ü":"Ue"}
COMPANY_TAILS = [
    r"\be\.?\s?v\.?\b", r"\bg\s*gmbh\b", r"\bgmbh\b", r"\bev\b", r"\bv\b",
    r"\bag\b", r"\bverein\b", r"\bgemeinn(ü|u)tzige?\b",
]
ADDR_ABBR = [(r"\bstraße\b","strasse"), (r"\bstr\.\b","strasse")]

def strip_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.map(lambda c: str(c).strip())
    return df

def _deumlaut(s: str) -> str:
    for k,v in UML.items():
        s = s.replace(k, v)
    s = "".join(c for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c))
    return s

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def norm_zip(z) -> str:
    if pd.isna(z): return ""
    z_str = re.sub(r"\D", "", str(z))
    return z_str if z_str else ""

def norm_name(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = _deumlaut(str(s)).lower()
    s = s.replace('"'," ").replace("’","'").replace("´","'")
    s = re.sub(r"[^\w\s\-&']", " ", s)
    for pat in COMPANY_TAILS:
        s = re.sub(pat, " ", s)
    return norm_space(s)

def norm_street(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = _deumlaut(str(s)).lower()
    for a,b in ADDR_ABBR:
        s = re.sub(a, b, s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    return norm_space(s)

def norm_city(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = _deumlaut(str(s)).lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    return norm_space(s)

def base_domain(d: str) -> str:
    if d is None or (isinstance(d, float) and pd.isna(d)): 
        return ""
    s = str(d).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    if "@" in s:
        s = s.split("@", 1)[1]
    parts = [p for p in s.split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else s

def long_tokens(nm: str) -> set:
    toks = [t for t in re.split(r"\W+", (nm or "")) if t]
    return {t for t in toks if len(t) >= 4}

# ------------- Load and prepare data -------------
def load_facilities(filename: str) -> pd.DataFrame:
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name="Facilities")
    df = strip_headers(df)
    df['_source'] = 'Facility'
    df['_source_id'] = 'F' + df.index.astype(str)
    df['_name'] = df.get('name', pd.Series([''] * len(df)))
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('adresse_plz', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('adresse_strasse', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('adresse_ort', pd.Series([''] * len(df))).apply(norm_city)
    # exact keys
    df['_num_awonr'] = df.get('carrier_id', pd.Series([''] * len(df))).astype(str).str.replace(r'\D','', regex=True)
    df['_domain_base'] = df.get('adresse_email', pd.Series([''] * len(df))).apply(base_domain)
    return df

def load_associations(filename: str) -> pd.DataFrame:
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name="Associations")
    df = strip_headers(df)
    df['_source'] = 'Association'
    df['_source_id'] = 'A' + df.index.astype(str)
    df['_name'] = df.get('name', pd.Series([''] * len(df)))
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('adresse_plz', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('adresse_strasse', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('adresse_ort', pd.Series([''] * len(df))).apply(norm_city)
    # exact keys
    df['_num_awonr'] = df.get('nummer', pd.Series([''] * len(df))).astype(str).str.replace(r'\D','', regex=True)
    df['_domain_base'] = df.get('adresse_email', pd.Series([''] * len(df))).apply(base_domain)
    return df

def load_legal(filename: str) -> pd.DataFrame:
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
    df = strip_headers(df)
    df['_source'] = 'LegalEntity'
    df['_source_id'] = 'L' + df.index.astype(str)
    # best name
    name_cols = [
        "Name der Körperschaft (lt. Handels- oder Vereinsregister)",
        "Trägername",
        "Name der Körperschaft",
        "Name",
    ]
    name = pd.Series([""]*len(df))
    for c in name_cols:
        if c in df.columns:
            name = name.where(name.astype(str).str.len() > 0, df[c].astype(str))
    df['_name'] = name
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('PLZ', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('Straße + Hausnr.', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('Ort', pd.Series([''] * len(df))).apply(norm_city)
    # exact keys
    awo1 = df.get('Nummer (AWO)', pd.Series([''] * len(df))).astype(str).str.replace(r"\D","", regex=True)
    awo2 = df.get('Träger-Id (AWO)', pd.Series([''] * len(df))).astype(str).str.replace(r"\D","", regex=True)
    df['_num_awonr'] = awo1.where(awo1 != "", awo2)
    df['_domain_base'] = df.get('Website', pd.Series([''] * len(df))).apply(base_domain)
    return df

def load_domains(filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(filename)
    else:
        xls = pd.ExcelFile(filename)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    df = strip_headers(df)
    df['_source'] = 'AWODomain'
    df['_source_id'] = 'D' + df.index.astype(str)
    df['_name'] = df.get('name', pd.Series([''] * len(df)))
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('plz', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('strasse', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('ort', pd.Series([''] * len(df))).apply(norm_city)
    # exact keys
    dom = df.get('domain', pd.Series([''] * len(df))).apply(base_domain)
    em_dom = df.get('email', pd.Series([''] * len(df))).apply(base_domain)
    df['_domain_base'] = dom.where(dom != "", em_dom)
    df['_num_awonr'] = ""  # not present
    return df

# ------------- Matching logic -------------
class EntityMatcher:
    def __init__(self):
        self.match_log = []

    def log_match(self, entity1_id, entity2_id, reason, score=None):
        entry = {'entity1': entity1_id, 'entity2': entity2_id, 'reason': reason}
        if score is not None:
            entry['score'] = score
        self.match_log.append(entry)

    def is_exact_name_match(self, name1, name2):
        if not name1 or not name2:
            return False
        if len(name1) < 3 or len(name2) < 3:
            return False
        return name1 == name2

    def is_fuzzy_name_match(self, name1, name2, threshold=FUZZY_NAME_MIN):
        if not name1 or not name2:
            return False, 0
        if len(name1) < 3 or len(name2) < 3:
            return False, 0
        # require a shared long token before scoring (cuts false positives)
        if len(long_tokens(name1).intersection(long_tokens(name2))) == 0:
            return False, 0
        score = sim(name1, name2)
        return score >= threshold, score

    def is_address_match(self, plz1, street1, plz2, street2):
        if not plz1 or not plz2:
            return False, 0
        if plz1 != plz2:
            return False, 0
        if not street1 or not street2:
            return False, 0
        if len(street1) < 3 or len(street2) < 3:
            return False, 0
        score = sim(street1, street2)
        return score >= FUZZY_STREET_MIN, score

    def should_merge(self, row1, row2):
        name1 = row1.get('_name_norm', '')
        name2 = row2.get('_name_norm', '')
        plz1 = row1.get('_plz', '')
        plz2 = row2.get('_plz', '')
        street1 = row1.get('_street_norm', '')
        street2 = row2.get('_street_norm', '')

        # NEW: exact AWO number
        n1, n2 = row1.get('_num_awonr', ''), row2.get('_num_awonr', '')
        if n1 and n2 and n1 == n2:
            return True, 'exact_awonr', 100

        # NEW: exact base domain
        d1, d2 = row1.get('_domain_base', ''), row2.get('_domain_base', '')
        if d1 and d2 and d1 == d2:
            return True, 'exact_domain', 100

        # Exact name
        if self.is_exact_name_match(name1, name2):
            return True, 'exact_name', 100

        # Address + name
        addr_match, addr_score = self.is_address_match(plz1, street1, plz2, street2)
        if addr_match:
            name_match, name_score = self.is_fuzzy_name_match(name1, name2, threshold=80)
            if name_match:
                return True, 'address_plus_name', int((addr_score + name_score) / 2)

        # Fuzzy name (conservative)
        fuzzy_match, fuzzy_score = self.is_fuzzy_name_match(name1, name2, threshold=FUZZY_NAME_MIN)
        if fuzzy_match:
            return True, 'fuzzy_name', fuzzy_score

        return False, None, 0

# ------------- Entity clustering -------------
class EntityCluster:
    """Represents a cluster of entities that are the same"""
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.entities = []  # List of (source, row_dict)

    def add_entity(self, source, row_dict):
        self.entities.append((source, row_dict))

    @property
    def num_sources(self):
        return len(self.entities)

    def has_source(self, source):
        return any(s == source for s, _ in self.entities)

    def _canonical_name(self) -> str:
        # Prefer Association > LegalEntity > Facility > AWODomain; longest within the chosen source
        order = {'Association': 3, 'LegalEntity': 2, 'Facility': 1, 'AWODomain': 0}
        best = (-1, -1, "")  # (rank:int, len:int, name:str) — all comparable
    
        for src, r in self.entities:
            nm = r.get('_name', '')
            # normalize and skip empties / NaNs
            if nm is None:
                continue
            nm = str(nm).strip()
            if not nm or nm.lower() == "nan":
                continue
    
            rank = int(order.get(src, -1))
            cand = (rank, len(nm), nm)
            if cand > best:
                best = cand
    
        return best[2]

    def _join_unique(self, series_like):
        vals = [str(v) for v in series_like if str(v).strip() and str(v).lower() != "nan"]
        if not vals:
            return ""
        return " | ".join(sorted(set(vals)))

    def to_merged_row(self):
        """Create a single row with all data from all sources in the requested format"""
        merged = {
            'EntityID': self.cluster_id,
            'IsFacility': False,
            'IsAssociation': False,
            'IsLegalEntity': False,
            'IsAWODomain': False,
            'EntityName': '',
            # Names per source
            'F_names': '',
            'A_names': '',
            'L_names': '',
            'D_names': '',
            # tail columns to be appended later
            'Members': '',
            'N_members': 0,
        }

        # Flags
        for source, _ in self.entities:
            if source == 'Facility':
                merged['IsFacility'] = True
            elif source == 'Association':
                merged['IsAssociation'] = True
            elif source == 'LegalEntity':
                merged['IsLegalEntity'] = True
            elif source == 'AWODomain':
                merged['IsAWODomain'] = True

        # Canonical name
        merged['EntityName'] = self._canonical_name()

        # Aggregate names per source (unique, pipe-separated)
        F_names = [r.get('_name','') for s, r in self.entities if s == 'Facility']
        A_names = [r.get('_name','') for s, r in self.entities if s == 'Association']
        L_names = [r.get('_name','') for s, r in self.entities if s == 'LegalEntity']
        D_names = [r.get('_name','') for s, r in self.entities if s == 'AWODomain']
        merged['F_names'] = self._join_unique(F_names)
        merged['A_names'] = self._join_unique(A_names)
        merged['L_names'] = self._join_unique(L_names)
        merged['D_names'] = self._join_unique(D_names)

        # Merge ALL original attributes with prefixes F_/A_/L_/D_
        prefix_map = {'Facility': 'F_', 'Association': 'A_', 'LegalEntity': 'L_', 'AWODomain': 'D_'}
        prefixed_values = defaultdict(list)
        for source, row_dict in self.entities:
            pref = prefix_map[source]
            for col, val in row_dict.items():
                if col.startswith('_'):
                    continue
                prefixed_values[pref + col].append(val)

        for fullcol, vals in prefixed_values.items():
            merged[fullcol] = self._join_unique(vals)

        # Members & counts
        members = [f"{s}:{r.get('_source_id','')}" for s, r in self.entities]
        merged['Members'] = " | ".join(sorted(members))
        merged['N_members'] = len(self.entities)

        return merged

def deduplicate_entities(facilities, associations, legal, domains):
    """Main deduplication logic"""
    matcher = EntityMatcher()

    # Prepare all entities
    all_entities = []
    for df, source in [(facilities, 'Facility'),
                       (associations, 'Association'),
                       (legal, 'LegalEntity'),
                       (domains, 'AWODomain')]:
        for _, row in df.iterrows():
            all_entities.append((source, row.to_dict()))

    print(f"Total entities to process: {len(all_entities)}")

    # Union-Find for clustering
    parent = list(range(len(all_entities)))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare pairs and merge if match (cross-source only)
    print("Finding matches...")
    n = len(all_entities)
    matches_found = 0
    for i in range(n):
        if i % 100 == 0:
            print(f"  Processed {i}/{n} entities...")
        source1, row1 = all_entities[i]
        for j in range(i + 1, n):
            source2, row2 = all_entities[j]
            if source1 == source2:
                continue
            should_merge, reason, score = matcher.should_merge(row1, row2)
            if should_merge:
                union(i, j)
                matcher.log_match(
                    f"{source1}:{row1.get('_source_id','')}",
                    f"{source2}:{row2.get('_source_id','')}",
                    reason,
                    score
                )
                matches_found += 1

    print(f"  Found {matches_found} matches")

    # Build clusters
    print("Building entity clusters...")
    clusters_dict = defaultdict(lambda: EntityCluster(0))
    for i in range(n):
        cluster_id = find(i)
        if clusters_dict[cluster_id].cluster_id == 0:
            clusters_dict[cluster_id].cluster_id = len(clusters_dict)
        source, row_dict = all_entities[i]
        clusters_dict[cluster_id].add_entity(source, row_dict)

    clusters = list(clusters_dict.values())
    print(f"  Created {len(clusters)} unique entities")

    return clusters, matcher.match_log

# ------------- Output generation -------------
def create_output_dataframe(clusters: list[EntityCluster]) -> pd.DataFrame:
    """Create final output dataframe in the requested column order."""
    rows = []

    # Keep your original ordering logic if needed; otherwise, stable by cluster id
    sorted_clusters = sorted(clusters, key=lambda c: c.cluster_id)
    for cluster in sorted_clusters:
        rows.append(cluster.to_merged_row())

    df = pd.DataFrame(rows)

    # Find all prefixed attribute columns
    f_cols = sorted([c for c in df.columns if c.startswith('F_')])
    a_cols = sorted([c for c in df.columns if c.startswith('A_')])
    l_cols = sorted([c for c in df.columns if c.startswith('L_')])
    d_cols = sorted([c for c in df.columns if c.startswith('D_')])

    # Exact order required
    ordered_cols = [
        'EntityID', 'IsFacility', 'IsAssociation', 'IsLegalEntity', 'IsAWODomain',
        'EntityName', 'F_names', 'A_names', 'L_names', 'D_names',
        *f_cols, *a_cols, *l_cols, *d_cols,
        'Members', 'N_members'
    ]
    # Keep only existing cols (in case some are empty)
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    return df[ordered_cols]

def write_match_logs(match_log, facilities, associations, legal, domains):
    """Write detailed match logs"""
    exact_matches = [m for m in match_log if m['reason'] in ('exact_name','exact_awonr','exact_domain')]
    address_matches = [m for m in match_log if m['reason'] == 'address_plus_name']
    fuzzy_matches = [m for m in match_log if m['reason'] == 'fuzzy_name']

    def write_log(filepath, matches, title):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n{title}\n{'='*80}\n\n")
            f.write(f"Total matches: {len(matches)}\n\n")
            for i, match in enumerate(matches, 1):
                f.write(f"\n{'-'*80}\n")
                f.write(f"Match #{i}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Entity 1: {match['entity1']}\n")
                f.write(f"Entity 2: {match['entity2']}\n")
                f.write(f"Reason: {match['reason']}\n")
                if 'score' in match:
                    f.write(f"Score: {match['score']}\n")

    write_log(OUT_TXT_EXACT, exact_matches, "EXACT MATCHES (AWO number / domain / name)")
    write_log(OUT_TXT_ADDR, address_matches, "ADDRESS + NAME MATCHES")
    write_log(OUT_TXT_FUZZY, fuzzy_matches, "FUZZY NAME MATCHES")

# ------------- Main -------------
def main():
    print("="*80)
    print("AWO ENTITY DEDUPLICATION")
    print("="*80)

    print("\n[1/5] Loading data...")
    facilities = _limit(load_facilities(IN_FAC_ASS), MAX_PER_SOURCE)
    associations = _limit(load_associations(IN_FAC_ASS), MAX_PER_SOURCE)
    legal = _limit(load_legal(IN_LEGAL), MAX_PER_SOURCE)
    domains = _limit(load_domains(IN_DOMAINS), MAX_PER_SOURCE)

    print(f"  [TEST MODE] Limited to <= {MAX_PER_SOURCE} per source:"
      f" F={len(facilities)}, A={len(associations)}, L={len(legal)}, D={len(domains)}")

    print(f"  ✓ Facilities: {len(facilities)}")
    print(f"  ✓ Associations: {len(associations)}")
    print(f"  ✓ Legal Entities: {len(legal)}")
    print(f"  ✓ AWO Domains: {len(domains)}")

    print("\n[2/5] Deduplicating entities...")
    clusters, match_log = deduplicate_entities(facilities, associations, legal, domains)

    print("\n[3/5] Creating output dataframe...")
    output_df = create_output_dataframe(clusters)

    print("\n[4/5] Writing Excel file...")
    output_df.to_excel(OUT_XLSX, index=False, engine='openpyxl')
    print(f"  ✓ {OUT_XLSX}")

    print("\n[5/5] Writing match logs...")
    write_match_logs(match_log, facilities, associations, legal, domains)
    print(f"  ✓ {OUT_TXT_EXACT}")
    print(f"  ✓ {OUT_TXT_ADDR}")
    print(f"  ✓ {OUT_TXT_FUZZY}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total input records: {len(facilities) + len(associations) + len(legal) + len(domains)}")
    print(f"Unique entities: {len(clusters)}")
    print(f"Entities with multiple sources: {len([c for c in clusters if c.num_sources > 1])}")
    print(f"\nBreakdown:")
    print(f"  Only in Facility: {len([c for c in clusters if c.has_source('Facility') and c.num_sources == 1])}")
    print(f"  Only in Association: {len([c for c in clusters if c.has_source('Association') and c.num_sources == 1])}")
    print(f"  Only in LegalEntity: {len([c for c in clusters if c.has_source('LegalEntity') and c.num_sources == 1])}")
    print(f"  Only in AWODomain: {len([c for c in clusters if c.has_source('AWODomain') and c.num_sources == 1])}")
    print("="*80)

if __name__ == "__main__":
    main()
