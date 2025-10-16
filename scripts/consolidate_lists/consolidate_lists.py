#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a deduplicated master list of AWO entities across 4 sources by
STRICT (name + full address) equality. No AWO number, no domains are used
for merging to avoid over-merging generic names.

Output Excel:
- Sheet "Entities": EntityID, flags, canonical name, per-source name lists,
  all source columns prefixed (F_/A_/L_/D_), Members, N_members
- Sheets with unique (Name + Address) pairs and counts per source:
  "Facilities_unique", "Associations_unique", "Legal_unique", "Domains_unique"
- NEW: "Name_Normalization_Debug" - shows which original names merged
- NEW: "Address_Normalization_Debug" - shows which original addresses merged
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from unidecode import unidecode
import pandas as pd

# -------------------- Config --------------------
ROOT = Path.cwd()

IN_FAC_ASS = ROOT / "data" / "input" / "2025_09_16_Einrichtunsdatenbank_Export_descriptions_final.xlsx"
IN_LEGAL   = ROOT / "data" / "input" / "2026_09_18_JurPers_Fragebogen2025_descriptions_final.xlsx"
IN_DOMAIN  = ROOT / "data" / "input" / "AWO_domains_with_impressum_data.xlsx"

OUT_DIR = ROOT / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUT_DIR / "deduplicated_entities.xlsx"

# For quick test runs; set to None for full data
MAX_PER_SOURCE = None  # e.g., 300

FLAG_COLS = ["IsFacility", "IsAssociation", "IsLegalEntity", "IsAWODomain"]

# -------------------- Helpers: normalization --------------------
UML = str.maketrans({"ä":"ae","ö":"oe","ü":"ue","Ä":"Ae","Ö":"Oe","Ü":"Ue","ß":"ss"})
# token aliases
_NAME_ALIASES = {
    "arbeiterwohlfahrt": "awo",
    "ortsverein": "ov",
    "kreisverband": "kv",
    "bezirksverband": "bv",
    "landesverband": "lv",
    "stadtverband": "stv",
    "unterbezirk": "ub",
    "regionalverband": "rv",
}
# legal/corporate forms to drop entirely
_LEGAL_FORM_RE = re.compile(
    r"""
    \b(
        ggmbh|gmbh|mbh|ag|ug|kgaa|kg|ohg|eg|ev|e\.?\s*v\.?|
        stiftung|verein|gag|eigenbetrieb
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)
# generic non-identifying words we often want to drop in names
# (keep this small and careful)
_GENERIC_RE = re.compile(
    r"\b(gemeinnuetzig(e|er|en)?|gemeinnützig(e|er|en)?)\b",
    re.IGNORECASE
)

def _to_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def norm_zip(z) -> str:
    z = _to_str(z)
    z = re.sub(r"\D", "", z)
    return z

def norm_city(s: str) -> str:
    s = _to_str(s).translate(UML).lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    return norm_space(s)

def norm_street(s: str) -> str:
    s = _to_str(s).translate(UML).lower()
    # unify Straße variants / abbreviations
    s = s.replace("straße", "strasse")
    s = re.sub(r"\bstr\.\b", "strasse", s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    return norm_space(s)

def norm_name(x: str) -> str:
    if x is None:
        return ""
    t = str(x).strip()
    if not t:
        return ""

    # lower + german transliteration (ae/oe/ue/ss)
    t = t.lower().translate(UML)

    # normalize punctuation
    t = _clean_punct(t)

    # --- collapse split legal abbreviations BEFORE tokenizing ---
    # handles "e v", "e. v.", "g g m b h", "g.m.b.h.", etc.
    t = re.sub(r"\be\s*\.?\s*v\b", "ev", t)
    t = re.sub(r"\bg\s*\.?\s*g\s*\.?\s*m\s*\.?\s*b\s*\.?\s*h\b", "ggmbh", t)
    t = re.sub(r"\bg\s*\.?\s*m\s*\.?\s*b\s*\.?\s*h\b", "gmbh", t)
    t = re.sub(r"\bm\s*\.?\s*b\s*\.?\s*h\b", "mbh", t)
    # ------------------------------------------------------------

    # tokenize, alias, drop legal forms & generic bits
    out_tokens = []
    for tok in t.split():
        if _LEGAL_FORM_RE.fullmatch(tok):
            continue
        if _GENERIC_RE.fullmatch(tok):
            continue
        tok = _NAME_ALIASES.get(tok, tok)
        out_tokens.append(tok)

    if not out_tokens:
        return ""

    # remove consecutive duplicates while keeping order
    seen = set()
    dedup = []
    for tok in out_tokens:
        if tok not in seen:
            dedup.append(tok)
            seen.add(tok)

    return re.sub(r"\s+", " ", " ".join(dedup)).strip()

def make_addr_key(zip_, city, street) -> str:
    z = norm_zip(zip_)
    c = norm_city(city)
    st = norm_street(street)
    # Require all parts; if anything is missing, return empty => no merging
    if not z or not c or not st:
        return ""
    return f"{z}|{c}|{st}"

def strip_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _clean_punct(s: str) -> str:
    # unify punctuation/dashes/& and remove the rest
    s = s.replace("&", " und ")
    s = re.sub(r"[–—\-_/|+.,:;()\"'´`""'‚]", " ", s)
    return s

# -------------------- Loaders --------------------
def load_facilities(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    df = pd.read_excel(xls, sheet_name="Facilities")
    df = strip_headers(df)

    out = df.copy()
    out["_source"] = "Facility"
    out["_source_id"] = "F" + out.index.astype(str)

    out["_name"] = out.get("name", "").astype(str)
    out["_name_orig"] = out["_name"]  # Keep original
    out["_name_norm"] = out["_name"].apply(norm_name)

    out["_zip_orig"] = out.get("adresse_plz", "").astype(str)
    out["_city_orig"] = out.get("adresse_ort", "").astype(str)
    out["_street_orig"] = out.get("adresse_strasse", "").astype(str)
    
    out["_zip"] = out["_zip_orig"].apply(norm_zip)
    out["_city_norm"] = out["_city_orig"].apply(norm_city)
    out["_street_norm"] = out["_street_orig"].apply(norm_street)
    out["_addr_key"] = out.apply(lambda r: make_addr_key(r["_zip"], r["_city_orig"], r["_street_orig"]), axis=1)
    return out


def load_associations(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    df = pd.read_excel(xls, sheet_name="Associations")
    df = strip_headers(df)

    out = df.copy()
    out["_source"] = "Association"
    out["_source_id"] = "A" + out.index.astype(str)

    out["_name"] = out.get("name", "").astype(str)
    out["_name_orig"] = out["_name"]
    out["_name_norm"] = out["_name"].apply(norm_name)

    out["_zip_orig"] = out.get("adresse_plz", "").astype(str)
    out["_city_orig"] = out.get("adresse_ort", "").astype(str)
    out["_street_orig"] = out.get("adresse_strasse", "").astype(str)
    
    out["_zip"] = out["_zip_orig"].apply(norm_zip)
    out["_city_norm"] = out["_city_orig"].apply(norm_city)
    out["_street_norm"] = out["_street_orig"].apply(norm_street)
    out["_addr_key"] = out.apply(lambda r: make_addr_key(r["_zip"], r["_city_orig"], r["_street_orig"]), axis=1)
    return out


def _read_biggest_sheet(excel_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    best_df, best_rows = None, -1
    for sh in xls.sheet_names:
        try:
            df_try = pd.read_excel(xls, sheet_name=sh)
            df_try = df_try.dropna(how="all")
            if len(df_try) > best_rows:
                best_rows = len(df_try)
                best_df = df_try
        except Exception:
            continue
    return strip_headers(best_df if best_df is not None else pd.DataFrame())


def load_legal(xlsx_path: Path) -> pd.DataFrame:
    df = _read_biggest_sheet(xlsx_path)

    out = df.copy()
    out["_source"] = "LegalEntity"
    out["_source_id"] = "L" + out.index.astype(str)

    # choose best name column
    name_cols = [
        "Name der Körperschaft (lt. Handels- oder Vereinsregister)",
        "Trägername",
        "Name der Körperschaft",
        "Name",
    ]
    name = pd.Series([""] * len(out), index=out.index, dtype=object)
    for c in name_cols:
        if c in out.columns:
            name = name.where(name.astype(str).str.len() > 0, out[c].astype(str))
    out["_name"] = name
    out["_name_orig"] = out["_name"]
    out["_name_norm"] = out["_name"].apply(norm_name)

    # address
    out["_zip_orig"] = out.get("PLZ", "").astype(str)
    out["_city_orig"] = out.get("Ort", "").astype(str)
    out["_street_orig"] = out.get("Straße + Hausnr.", "").astype(str)
    
    out["_zip"] = out["_zip_orig"].apply(norm_zip)
    out["_city_norm"] = out["_city_orig"].apply(norm_city)
    out["_street_norm"] = out["_street_orig"].apply(norm_street)
    out["_addr_key"] = out.apply(lambda r: make_addr_key(r["_zip"], r["_city_orig"], r["_street_orig"]), axis=1)
    return out


def load_domains(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    df = strip_headers(df)

    out = df.copy()
    out["_source"] = "AWODomain"
    out["_source_id"] = "D" + out.index.astype(str)

    nm_col = "name" if "name" in out.columns else ("Name" if "Name" in out.columns else None)
    out["_name"] = out[nm_col].astype(str) if nm_col else ""
    out["_name_orig"] = out["_name"]
    out["_name_norm"] = out["_name"].apply(norm_name)

    # street = strasse + hausnummer
    out["_street_orig"] = ((out.get("strasse","").astype(str) + " " + out.get("hausnummer","").astype(str)).str.strip())
    out["_zip_orig"] = out.get("plz", "").astype(str)
    out["_city_orig"] = out.get("ort", "").astype(str)
    
    out["_zip"] = out["_zip_orig"].apply(norm_zip)
    out["_city_norm"] = out["_city_orig"].apply(norm_city)
    out["_street_norm"] = out["_street_orig"].apply(norm_street)
    out["_addr_key"] = out.apply(lambda r: make_addr_key(r["_zip"], r["_city_orig"], r["_street_orig"]), axis=1)
    return out


def _limit(df: pd.DataFrame, max_n):
    if not max_n:
        return df
    n = min(int(max_n), len(df))
    return df.sample(n=n, random_state=42)


# -------------------- Clustering: strict name+address --------------------
class EntityCluster:
    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        self.entities = []  # list of (src, row_dict)

    def add(self, src: str, row: dict):
        self.entities.append((src, row))

    def has_source(self, src: str) -> bool:
        return any(s == src for s, _ in self.entities)

    def _pick_canonical_name(self) -> str:
        # Prefer Association > Legal > Facility > Domain; longest string in that tier
        order = {"Association": 3, "LegalEntity": 2, "Facility": 1, "AWODomain": 0}
        best = (-1, -1, "")
        for s, r in self.entities:
            nm = str(r.get("_name","")).strip()
            if not nm:
                continue
            cand = (order.get(s, -1), len(nm), nm)
            if cand > best:
                best = cand
        return best[2]

    @staticmethod
    def _join_unique(iterable):
        vals = [str(v).strip() for v in iterable if str(v).strip()]
        if not vals:
            return ""
        return " | ".join(sorted(set(vals)))

    def to_row(self) -> dict:
        row = {
            "EntityID": self.cluster_id,
            "IsFacility": self.has_source("Facility"),
            "IsAssociation": self.has_source("Association"),
            "IsLegalEntity": self.has_source("LegalEntity"),
            "IsAWODomain": self.has_source("AWODomain"),
            "EntityName": self._pick_canonical_name(),
            "F_names": self._join_unique(r.get("_name","") for s,r in self.entities if s=="Facility"),
            "A_names": self._join_unique(r.get("_name","") for s,r in self.entities if s=="Association"),
            "L_names": self._join_unique(r.get("_name","") for s,r in self.entities if s=="LegalEntity"),
            "D_names": self._join_unique(r.get("_name","") for s,r in self.entities if s=="AWODomain"),
            "Members": "",
            "N_members": len(self.entities),
        }

        # copy all visible columns with prefixes
        prefix = {"Facility":"F_", "Association":"A_", "LegalEntity":"L_", "AWODomain":"D_"}
        buckets = defaultdict(list)
        for s, r in self.entities:
            pref = prefix[s]
            for col, val in r.items():
                if col.startswith("_"):  # skip internal fields
                    continue
                buckets[pref + col].append(val)

        for k, vals in buckets.items():
            row[k] = self._join_unique(vals)

        members = [f"{s}:{r.get('_source_id','')}" for s, r in self.entities]
        row["Members"] = " | ".join(sorted(members))
        return row


def build_clusters(fac: pd.DataFrame, ass: pd.DataFrame, leg: pd.DataFrame, dom: pd.DataFrame):
    """
    Strict clustering:
      key = (name_norm, addr_key)
      - If either part is empty, the record becomes its own entity (no merge).
      - Otherwise, any records (even within same source) with the SAME key merge.
      - NO AWO number, NO domain-based merges.
    """
    all_rows = []
    for df, src in [(fac,"Facility"), (ass,"Association"), (leg,"LegalEntity"), (dom,"AWODomain")]:
        for _, r in df.iterrows():
            d = r.to_dict()
            d["_source"] = src
            all_rows.append(d)

    key_to_clusterid = {}
    clusters = []
    next_id = 1

    for d in all_rows:
        name_key = d.get("_name_norm","")
        addr_key = d.get("_addr_key","")
        if name_key and addr_key:
            k = (name_key, addr_key)
            if k not in key_to_clusterid:
                key_to_clusterid[k] = next_id
                clusters.append(EntityCluster(next_id))
                next_id += 1
            cid = key_to_clusterid[k]
        else:
            # no full address or empty name -> unique singleton cluster
            cid = next_id
            clusters.append(EntityCluster(next_id))
            next_id += 1

        # add to cluster with id cid
        clusters[cid-1].add(d["_source"], d)

    return clusters


# -------------------- Normalization Debug Sheets --------------------
def name_normalization_debug_sheet(fac, ass, leg, dom, clusters) -> pd.DataFrame:
    """
    Show which original names collapsed into the same normalized name.
    """
    # Build entity_id mapping
    source_to_entity = {}
    for cluster in clusters:
        eid = cluster.cluster_id
        for src, row in cluster.entities:
            sid = row.get("_source_id", "")
            if sid:
                source_to_entity[(src, sid)] = eid
    
    # Collect all name records
    records = []
    for df, src_name in [(fac, "Facility"), (ass, "Association"), 
                          (leg, "LegalEntity"), (dom, "AWODomain")]:
        for _, row in df.iterrows():
            name_orig = str(row.get("_name_orig", "")).strip()
            name_norm = str(row.get("_name_norm", "")).strip()
            source_id = row.get("_source_id", "")
            entity_id = source_to_entity.get((src_name, source_id), None)
            
            if name_norm:  # Only include non-empty normalized names
                records.append({
                    "name_norm": name_norm,
                    "name_orig": name_orig,
                    "source": src_name,
                    "entity_id": entity_id
                })
    
    df_records = pd.DataFrame(records)
    
    if len(df_records) == 0:
        return pd.DataFrame()
    
    # Aggregate by normalized name
    def _join_unique(s):
        vals = [str(x).strip() for x in s if str(x).strip() and str(x) != "nan"]
        return " | ".join(sorted(set(vals)))
    
    def _join_comma(s):
        vals = [str(int(x)) for x in s if pd.notna(x)]
        return ", ".join(sorted(set(vals)))
    
    def _count_unique(s):
        return len(set([str(x).strip() for x in s if str(x).strip() and str(x) != "nan"]))
    
    agg = df_records.groupby("name_norm", as_index=False).agg(
        N_Original_Variants=("name_orig", _count_unique),
        Original_Names=("name_orig", _join_unique),
        N_Entities=("entity_id", "nunique"),
        EntityIDs=("entity_id", _join_comma),
        Sources=("source", _join_unique)
    )
    
    # Only keep where there are multiple variants or multiple entities
    agg = agg[agg["N_Original_Variants"] > 1]
    
    # Sort by most variants first
    agg = agg.sort_values("N_Original_Variants", ascending=False).reset_index(drop=True)
    
    # Rename for clarity
    agg = agg.rename(columns={"name_norm": "Normalized_Name"})
    
    return agg


def address_normalization_debug_sheet(fac, ass, leg, dom, clusters) -> pd.DataFrame:
    """
    Show which original addresses collapsed into the same addr_key.
    """
    # Build entity_id mapping
    source_to_entity = {}
    for cluster in clusters:
        eid = cluster.cluster_id
        for src, row in cluster.entities:
            sid = row.get("_source_id", "")
            if sid:
                source_to_entity[(src, sid)] = eid
    
    # Collect all address records
    records = []
    for df, src_name in [(fac, "Facility"), (ass, "Association"), 
                          (leg, "LegalEntity"), (dom, "AWODomain")]:
        for _, row in df.iterrows():
            addr_key = str(row.get("_addr_key", "")).strip()
            zip_orig = str(row.get("_zip_orig", "")).strip()
            city_orig = str(row.get("_city_orig", "")).strip()
            street_orig = str(row.get("_street_orig", "")).strip()
            
            source_id = row.get("_source_id", "")
            entity_id = source_to_entity.get((src_name, source_id), None)
            
            if addr_key:  # Only include non-empty addr_keys
                full_addr = f"{zip_orig} {city_orig}, {street_orig}" if street_orig else f"{zip_orig} {city_orig}"
                records.append({
                    "addr_key": addr_key,
                    "full_addr_orig": full_addr,
                    "zip_orig": zip_orig,
                    "city_orig": city_orig,
                    "street_orig": street_orig,
                    "source": src_name,
                    "entity_id": entity_id
                })
    
    df_records = pd.DataFrame(records)
    
    if len(df_records) == 0:
        return pd.DataFrame()
    
    # Aggregate by addr_key
    def _join_unique(s):
        vals = [str(x).strip() for x in s if str(x).strip() and str(x) != "nan"]
        return " | ".join(sorted(set(vals)))
    
    def _join_comma(s):
        vals = [str(int(x)) for x in s if pd.notna(x)]
        return ", ".join(sorted(set(vals)))
    
    def _count_unique(s):
        return len(set([str(x).strip() for x in s if str(x).strip() and str(x) != "nan"]))
    
    agg = df_records.groupby("addr_key", as_index=False).agg(
        N_Original_Variants=("full_addr_orig", _count_unique),
        Original_Addresses_Full=("full_addr_orig", _join_unique),
        Original_ZIP_Variants=("zip_orig", _join_unique),
        Original_City_Variants=("city_orig", _join_unique),
        Original_Street_Variants=("street_orig", _join_unique),
        N_Entities=("entity_id", "nunique"),
        EntityIDs=("entity_id", _join_comma),
        Sources=("source", _join_unique)
    )
    
    # Only keep where there are multiple variants
    agg = agg[agg["N_Original_Variants"] > 1]
    
    # Sort by most variants first
    agg = agg.sort_values("N_Original_Variants", ascending=False).reset_index(drop=True)
    
    return agg


# -------------------- Unique (name+address) sheets --------------------
def unique_name_address_sheet(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Returns a dataframe with unique (Name, ZIP, City, Street) pairs and counts for the given source.
    Tries to show original (non-normalized) address columns where possible.
    """
    # pick display columns per source
    if source == "Facility":
        disp_name  = df.get("name", "")
        disp_zip   = df.get("adresse_plz", "")
        disp_city  = df.get("adresse_ort", "")
        disp_street= df.get("adresse_strasse", "")
    elif source == "Association":
        disp_name  = df.get("name", "")
        disp_zip   = df.get("adresse_plz", "")
        disp_city  = df.get("adresse_ort", "")
        disp_street= df.get("adresse_strasse", "")
    elif source == "LegalEntity":
        # legal names live in multiple cols; show the chosen _name
        disp_name  = df.get("_name", "")
        disp_zip   = df.get("PLZ", "")
        disp_city  = df.get("Ort", "")
        disp_street= df.get("Straße + Hausnr.", "")
    else:  # AWODomain
        disp_name  = df.get("_name", "")
        disp_zip   = df.get("plz", "")
        disp_city  = df.get("ort", "")
        disp_street= (df.get("strasse","").astype(str) + " " + df.get("hausnummer","").astype(str)).str.strip()

    tmp = pd.DataFrame({
        "Name": disp_name.astype(str),
        "ZIP":  disp_zip.astype(str),
        "City": disp_city.astype(str),
        "Street": disp_street.astype(str),
        "_name_norm": df["_name_norm"],
        "_addr_key":  df["_addr_key"],
    })

    # Group by normalized keys to ensure true uniqueness; pick representative display values
    grp = tmp.groupby(["_name_norm","_addr_key"], dropna=False, as_index=False).agg(
        Name=("Name","first"),
        ZIP=("ZIP","first"),
        City=("City","first"),
        Street=("Street","first"),
        Count=("Name","size"),
    )

    # Sort for readability
    grp = grp.sort_values(["Name","ZIP","City","Street"]).reset_index(drop=True)
    # Hide the normalized keys in the sheet? Keep them (useful for QA). Comment next line to keep:
    # grp = grp[["Name","ZIP","City","Street","Count"]]
    return grp

# -------------------- Unique (address) sheet --------------------
def address_name_collisions_sheet(fac, ass, leg, dom) -> pd.DataFrame:
    """
    For every full address key (_addr_key), collect ALL normalized names seen
    across sources. Keep only addresses that have >1 distinct normalized name.
    """
    base = ["_addr_key", "_zip", "_city_norm", "_street_norm", "_name_norm"]

    fac2 = fac[base].assign(_source="Facility")
    ass2 = ass[base].assign(_source="Association")
    leg2 = leg[base].assign(_source="LegalEntity")
    dom2 = dom[base].assign(_source="AWODomain")

    all_df = pd.concat([fac2, ass2, leg2, dom2], ignore_index=True)

    # keep only rows with a complete address key
    df = all_df[all_df["_addr_key"].astype(str).str.len() > 0].copy()

    # counts per source for each address
    src_counts = (
        df.groupby(["_addr_key", "_source"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    # aggregate names & basic address info
    def _uniq_names(s):
        vals = [str(x).strip() for x in s if str(x).strip()]
        return " | ".join(sorted(set(vals)))

    def _n_uniq(s):
        return len(set([str(x).strip() for x in s if str(x).strip()]))

    agg = (
        df.groupby("_addr_key", as_index=False)
          .agg(
              ZIP=("_zip", "first"),
              City_norm=("_city_norm", "first"),
              Street_norm=("_street_norm", "first"),
              Names_norm=("_name_norm", _uniq_names),
              N_names=("_name_norm", _n_uniq),
              N_records=("_name_norm", "size"),
          )
    )

    out = agg.merge(src_counts, on="_addr_key", how="left")
    out = out[out["N_names"] > 1]  # only addresses with multiple (normalized) names
    out = out.sort_values(["N_names", "N_records"], ascending=False).reset_index(drop=True)

    # nice column order (only if they exist)
    ordered = [
        "ZIP", "City_norm", "Street_norm",
        "N_names", "N_records",
        "Facility", "Association", "LegalEntity", "AWODomain",
        "Names_norm", "_addr_key",
    ]
    existing = [c for c in ordered if c in out.columns]
    return out[existing]

# -------------------- Summmary: overlap --------------------
def _normalize_flags(df, flag_cols=FLAG_COLS):
    """Coerce flags to boolean (True/False)."""
    for c in flag_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
        s = df[c]
        if s.dtype == bool:
            df[c] = s.fillna(False)
        else:
            df[c] = (
                s.astype(str)
                 .str.strip().str.upper()
                 .isin(["TRUE", "T", "1", "YES", "Y"])
            )
    return df

def print_overlap_report(df, flag_cols=FLAG_COLS, inclusive=False):
    """
    inclusive=False (default): exact combos only (e.g., 'Facility+Association' but not those also in LegalEntity).
    inclusive=True: 'at least' those flags (supersets allowed).
    """
    df = df.copy()
    _normalize_flags(df, flag_cols)

    # helper for nice labels
    def nice(name):
        return {
            "IsFacility":"Facilities",
            "IsAssociation":"Associations",
            "IsLegalEntity":"Legal Entities",
            "IsAWODomain":"AWO Domains",
        }[name]

    # ---- ALL FOUR ----
    mask_all4 = df[flag_cols].all(axis=1)
    print(f"Entries overlap in all 4: {int(mask_all4.sum())}")

    # ---- TRIPLES ----
    for trio in combinations(flag_cols, 3):
        if inclusive:
            mask = df[list(trio)].all(axis=1)  # allow the 4th to be either
        else:
            others = [c for c in flag_cols if c not in trio]
            mask = df[list(trio)].all(axis=1) & (~df[others].any(axis=1))
        label = ", ".join(nice(c) for c in trio)
        print(f"Entries overlap in {label}: {int(mask.sum())}")

    # ---- PAIRS ----
    for pair in combinations(flag_cols, 2):
        if inclusive:
            mask = df[list(pair)].all(axis=1)  # allow others
        else:
            others = [c for c in flag_cols if c not in pair]
            mask = df[list(pair)].all(axis=1) & (~df[others].any(axis=1))
        label = ", ".join(nice(c) for c in pair)
        print(f"Entries overlap in {label}: {int(mask.sum())}")

    # ---- SINGLES (only in X) ----
    for c in flag_cols:
        others = [x for x in flag_cols if x != c]
        mask = df[c] & (~df[others].any(axis=1))
        print(f"Entries found only in {nice(c)}: {int(mask.sum())}")

    # ---- NONE (sanity check) ----
    mask_none = ~df[flag_cols].any(axis=1)
    if mask_none.any():
        print(f"Entries with no source flag set: {int(mask_none.sum())}")

# -------------------- Output --------------------
def create_entities_dataframe(clusters) -> pd.DataFrame:
    rows = [c.to_row() for c in clusters]
    df = pd.DataFrame(rows)

    # order columns: fixed header, then prefixed blocks, then members
    fixed = [
        "EntityID","IsFacility","IsAssociation","IsLegalEntity","IsAWODomain",
        "EntityName","F_names","A_names","L_names","D_names"
    ]
    f_cols = sorted([c for c in df.columns if c.startswith("F_")])
    a_cols = sorted([c for c in df.columns if c.startswith("A_")])
    l_cols = sorted([c for c in df.columns if c.startswith("L_")])
    d_cols = sorted([c for c in df.columns if c.startswith("D_")])
    tail = ["Members","N_members"]

    ordered = [c for c in fixed if c in df.columns] + f_cols + a_cols + l_cols + d_cols + [c for c in tail if c in df.columns]
    return df[ordered]


def main():
    print("="*80)
    print("AWO Entity Dedup (STRICT name + address; no AWO number, no domains)")
    print("="*80)

    print("\n[1/5] Loading data...")
    fac = load_facilities(IN_FAC_ASS)
    ass = load_associations(IN_FAC_ASS)
    leg = load_legal(IN_LEGAL)
    dom = load_domains(IN_DOMAIN)

    if MAX_PER_SOURCE:
        fac = fac.sample(n=min(MAX_PER_SOURCE, len(fac)), random_state=42)
        ass = ass.sample(n=min(MAX_PER_SOURCE, len(ass)), random_state=42)
        leg = leg.sample(n=min(MAX_PER_SOURCE, len(leg)), random_state=42)
        dom = dom.sample(n=min(MAX_PER_SOURCE, len(dom)), random_state=42)

    print(f"  Facilities  : {len(fac)}")
    print(f"  Associations: {len(ass)}")
    print(f"  Legal       : {len(leg)}")
    print(f"  Domains     : {len(dom)}")

    print("\n[2/5] Clustering by (name_norm, addr_key)...")
    clusters = build_clusters(fac, ass, leg, dom)
    print(f"  Unique entities: {len(clusters)}")

    print("\n[3/5] Building normalization debug sheets...")
    df_name_debug = name_normalization_debug_sheet(fac, ass, leg, dom, clusters)
    df_addr_debug = address_normalization_debug_sheet(fac, ass, leg, dom, clusters)
    print(f"  Name normalization issues: {len(df_name_debug)}")
    print(f"  Address normalization issues: {len(df_addr_debug)}")

    print("\n[4/5] Building output dataframes...")
    df_entities = create_entities_dataframe(clusters)
    df_fac_u = unique_name_address_sheet(fac, "Facility")
    df_ass_u = unique_name_address_sheet(ass, "Association")
    df_leg_u = unique_name_address_sheet(leg, "LegalEntity")
    df_dom_u = unique_name_address_sheet(dom, "AWODomain")
    df_addr_name_collisions = address_name_collisions_sheet(fac, ass, leg, dom)

    print("\n[5/5] Writing Excel with all sheets...")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        df_entities.to_excel(xw, sheet_name="Entities", index=False)
        df_name_debug.to_excel(xw, sheet_name="Name_Normalization_Debug", index=False)
        df_addr_debug.to_excel(xw, sheet_name="Address_Normalization_Debug", index=False)
        df_fac_u.to_excel(xw, sheet_name="Facilities_unique", index=False)
        df_ass_u.to_excel(xw, sheet_name="Associations_unique", index=False)
        df_leg_u.to_excel(xw, sheet_name="Legal_unique", index=False)
        df_dom_u.to_excel(xw, sheet_name="Domains_unique", index=False)
        df_addr_name_collisions.to_excel(xw, sheet_name="Addr_Name_Collisions", index=False)


    print("\nSummary:")
    print_overlap_report(df_entities, inclusive=False)

    print(f"\n✓ Wrote {OUT_XLSX}")
    print("="*80)
    print("Done.")


if __name__ == "__main__":
    main()