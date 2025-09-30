#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AWO Entity Deduplication - Creates a master list of unique entities
Each row represents one unique entity with data from all sources where it was found
"""

import os
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ------------- Config -------------
# Name similarity threshold for fuzzy matching
FUZZY_NAME_MIN = 88

# Address matching requires PLZ match + street similarity
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
    if not s or pd.isna(s): return ""
    s = _deumlaut(str(s)).lower()
    s = s.replace('"'," ").replace("'","'")
    s = re.sub(r"[^\w\s\-&']", " ", s)
    for pat in COMPANY_TAILS:
        s = re.sub(pat, " ", s)
    return norm_space(s)

def norm_street(s: str) -> str:
    if not s or pd.isna(s): return ""
    s = _deumlaut(str(s)).lower()
    for a,b in ADDR_ABBR:
        s = re.sub(a, b, s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    return norm_space(s)

def norm_city(s: str) -> str:
    if not s or pd.isna(s): return ""
    s = _deumlaut(str(s)).lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    return norm_space(s)

# ------------- Load and prepare data -------------
def load_facilities(filename: str) -> pd.DataFrame:
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name="Facilities")
    df['_source'] = 'Facility'
    df['_source_id'] = 'F' + df.index.astype(str)
    df['_name'] = df.get('name', pd.Series([''] * len(df)))
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('adresse_plz', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('adresse_strasse', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('adresse_ort', pd.Series([''] * len(df))).apply(norm_city)
    return df

def load_associations(filename: str) -> pd.DataFrame:
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name="Associations")
    df['_source'] = 'Association'
    df['_source_id'] = 'A' + df.index.astype(str)
    df['_name'] = df.get('name', pd.Series([''] * len(df)))
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('adresse_plz', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('adresse_strasse', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('adresse_ort', pd.Series([''] * len(df))).apply(norm_city)
    return df

def load_legal(filename: str) -> pd.DataFrame:
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    df['_source'] = 'LegalEntity'
    df['_source_id'] = 'L' + df.index.astype(str)
    
    # Find best name column
    name_cols = [
        "Name der Körperschaft (lt. Handels- oder Vereinsregister)",
        "Trägername",
        "Name der Körperschaft",
        "Name",
    ]
    name = pd.Series([None]*len(df))
    for c in name_cols:
        if c in df.columns:
            name = name.fillna(df[c])
    df['_name'] = name.fillna('')
    df['_name_norm'] = df['_name'].apply(norm_name)
    
    df['_plz'] = df.get('PLZ', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('Straße + Hausnr.', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('Ort', pd.Series([''] * len(df))).apply(norm_city)
    return df

def load_domains(filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(filename)
    else:
        xls = pd.ExcelFile(filename)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    
    df['_source'] = 'AWODomain'
    df['_source_id'] = 'D' + df.index.astype(str)
    df['_name'] = df.get('name', pd.Series([''] * len(df)))
    df['_name_norm'] = df['_name'].apply(norm_name)
    df['_plz'] = df.get('plz', pd.Series([''] * len(df))).apply(norm_zip)
    df['_street_norm'] = df.get('strasse', pd.Series([''] * len(df))).apply(norm_street)
    df['_city_norm'] = df.get('ort', pd.Series([''] * len(df))).apply(norm_city)
    return df

# ------------- Matching logic -------------
class EntityMatcher:
    def __init__(self):
        self.match_log = []
        
    def log_match(self, entity1_id, entity2_id, reason, score=None):
        """Log a match with reason"""
        entry = {
            'entity1': entity1_id,
            'entity2': entity2_id,
            'reason': reason
        }
        if score:
            entry['score'] = score
        self.match_log.append(entry)
    
    def is_exact_name_match(self, name1, name2):
        """Check if normalized names match exactly (must be substantial)"""
        if not name1 or not name2:
            return False
        if len(name1) < 3 or len(name2) < 3:
            return False
        return name1 == name2
    
    def is_fuzzy_name_match(self, name1, name2, threshold=FUZZY_NAME_MIN):
        """Check if names are similar enough"""
        if not name1 or not name2:
            return False, 0
        if len(name1) < 3 or len(name2) < 3:
            return False, 0
        score = sim(name1, name2)
        return score >= threshold, score
    
    def is_address_match(self, plz1, street1, plz2, street2):
        """Check if addresses match: PLZ must be exact, street must be similar"""
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
        """Determine if two entities should be merged"""
        name1 = row1['_name_norm']
        name2 = row2['_name_norm']
        plz1 = row1['_plz']
        plz2 = row2['_plz']
        street1 = row1['_street_norm']
        street2 = row2['_street_norm']
        
        # Exact name match
        if self.is_exact_name_match(name1, name2):
            return True, 'exact_name', 100
        
        # Address match (PLZ + street)
        addr_match, addr_score = self.is_address_match(plz1, street1, plz2, street2)
        if addr_match:
            # If address matches, check if names are somewhat similar
            name_match, name_score = self.is_fuzzy_name_match(name1, name2, threshold=70)
            if name_match:
                return True, 'address_plus_name', (addr_score + name_score) / 2
        
        # Fuzzy name match
        fuzzy_match, fuzzy_score = self.is_fuzzy_name_match(name1, name2)
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
    
    def to_merged_row(self):
        """Create a single row with all data from all sources"""
        merged = {
            'cluster_id': self.cluster_id,
            'IsFacility': False,
            'IsAssociation': False,
            'IsLegalEntity': False,
            'IsAWODomain': False,
            'num_sources': len(self.entities),
            'sources': ', '.join(s for s, _ in self.entities)
        }
        
        # Set source flags
        for source, _ in self.entities:
            if source == 'Facility':
                merged['IsFacility'] = True
            elif source == 'Association':
                merged['IsAssociation'] = True
            elif source == 'LegalEntity':
                merged['IsLegalEntity'] = True
            elif source == 'AWODomain':
                merged['IsAWODomain'] = True
        
        # Merge data from each source
        for source, row_dict in self.entities:
            for col, val in row_dict.items():
                if col.startswith('_'):  # Skip internal columns
                    continue
                
                # Create source-specific column name
                col_name = f"{source}_{col}"
                merged[col_name] = val
        
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
        for idx, row in df.iterrows():
            all_entities.append((source, row.to_dict()))
    
    print(f"Total entities to process: {len(all_entities)}")
    
    # Union-Find structure for clustering
    parent = list(range(len(all_entities)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Compare all pairs and merge if they match
    print("Finding matches...")
    n = len(all_entities)
    matches_found = 0
    
    for i in range(n):
        if i % 100 == 0:
            print(f"  Processed {i}/{n} entities...")
        
        source1, row1 = all_entities[i]
        
        # Compare with all subsequent entities
        for j in range(i + 1, n):
            source2, row2 = all_entities[j]
            
            # Don't compare entities from the same source
            if source1 == source2:
                continue
            
            should_merge, reason, score = matcher.should_merge(row1, row2)
            
            if should_merge:
                union(i, j)
                matcher.log_match(
                    f"{source1}:{row1['_source_id']}",
                    f"{source2}:{row2['_source_id']}",
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
def create_output_dataframe(clusters):
    """Create final output dataframe"""
    rows = []
    
    # Sort clusters: Facilities first, then Associations, then AWODomain, then LegalEntity
    def cluster_sort_key(cluster):
        priority = 0
        if cluster.has_source('Facility'):
            priority = 0
        elif cluster.has_source('Association'):
            priority = 1
        elif cluster.has_source('AWODomain'):
            priority = 2
        elif cluster.has_source('LegalEntity'):
            priority = 3
        return priority, cluster.cluster_id
    
    sorted_clusters = sorted(clusters, key=cluster_sort_key)
    
    for cluster in sorted_clusters:
        rows.append(cluster.to_merged_row())
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: source flags first, then facility cols, association cols, etc.
    priority_cols = [
        'cluster_id', 'num_sources', 'sources',
        'IsFacility', 'IsAssociation', 'IsLegalEntity', 'IsAWODomain'
    ]
    
    facility_cols = [c for c in df.columns if c.startswith('Facility_')]
    association_cols = [c for c in df.columns if c.startswith('Association_')]
    legal_cols = [c for c in df.columns if c.startswith('LegalEntity_')]
    domain_cols = [c for c in df.columns if c.startswith('AWODomain_')]
    
    ordered_cols = priority_cols + facility_cols + association_cols + legal_cols + domain_cols
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    
    return df[ordered_cols]

def write_match_logs(match_log, facilities, associations, legal, domains):
    """Write detailed match logs"""
    
    # Group by match type
    exact_matches = [m for m in match_log if m['reason'] == 'exact_name']
    address_matches = [m for m in match_log if m['reason'] == 'address_plus_name']
    fuzzy_matches = [m for m in match_log if m['reason'] == 'fuzzy_name']
    
    def write_log(filepath, matches, title):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{title}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total matches: {len(matches)}\n\n")
            
            for i, match in enumerate(matches, 1):
                f.write(f"\n{'-'*80}\n")
                f.write(f"Match #{i}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Entity 1: {match['entity1']}\n")
                f.write(f"Entity 2: {match['entity2']}\n")
                f.write(f"Reason: {match['reason']}\n")
                if 'score' in match:
                    f.write(f"Score: {match['score']:.1f}\n")
    
    write_log(OUT_TXT_EXACT, exact_matches, "EXACT NAME MATCHES")
    write_log(OUT_TXT_ADDR, address_matches, "ADDRESS + NAME MATCHES")
    write_log(OUT_TXT_FUZZY, fuzzy_matches, "FUZZY NAME MATCHES")

# ------------- Main -------------
def main():
    print("="*80)
    print("AWO ENTITY DEDUPLICATION")
    print("="*80)
    
    print("\n[1/5] Loading data...")
    facilities = load_facilities(IN_FAC_ASS)
    associations = load_associations(IN_FAC_ASS)
    legal = load_legal(IN_LEGAL)
    domains = load_domains(IN_DOMAINS)
    
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