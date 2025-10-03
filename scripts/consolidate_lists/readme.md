# consolidate_lists

Merge the four AWO sources — **Facilities**, **Associations**, **Legal Entities**, **AWO Domains** — into a **single, deduplicated master list**.

- One row per unique entity (based on normalized `name + address`).
- Source flags: `IsFacility`, `IsAssociation`, `IsLegalEntity`, `IsAWODomain`.
- Includes overlap counts and per-source _unique name+address_ sheets.

## Inputs (expected columns)

- **Facilities**: Excel with sheet _Facilities_ (or similar); needs at least `name`, `adresse_strasse`, `adresse_plz`, `adresse_ort`.
- **Associations**: Excel with sheet _Associations_ (or similar); needs at least `name`, `adresse_strasse`, `adresse_plz`, `adresse_ort`.
- **Legal Entities**: Excel; best-effort name from `Name der Körperschaft (lt. Handels- oder Vereinsregister)` or `Trägername`; address from `Straße + Hausnr.`, `PLZ`, `Ort`.
- **AWO Domains**: Excel/CSV with at least `name` (or `domain` as fallback) and optional `strasse/hausnummer/plz/ort`.

> Column names are case-insensitive; the script performs header stripping and normalization.

## Output

An Excel file with (typical) sheets:

- `MasterEntities` — columns like `EntityID, IsFacility, IsAssociation, IsLegalEntity, IsAWODomain, EntityName, F_names, A_names, L_names, D_names, ...`.
- `Unique_Facility_NamesAddresses` — list of `(normalized_name, normalized_address)` combos that define Facility uniqueness.
- `Unique_Association_NamesAddresses` — same for Associations.
- `Unique_Legal_NamesAddresses` — same for Legal Entities.
- `Unique_Domain_NamesAddresses` — same for AWO Domains.
- `AddressToNames` — for each normalized address across _all_ sources, the set of normalized names seen.
- `OverlapSummary` — counts for all source-combination overlaps.

## Run

```bash
python consolidate_lists.py   --fac data/input/2025_09_16_Einrichtunsdatenbank_Export_descriptions_final.xlsx   --ass data/input/2026_09_18_JurPers_Fragebogen2025_descriptions_final.xlsx   --leg data/input/2026_09_18_JurPers_Fragebogen2025_descriptions_final.xlsx   --dom data/input/AWO_domains_with_impressum_data.xlsx   --out data/output/merged_entities.xlsx   --limit 0
```

### Key options

- `--limit` _(int, default 0)_: limit rows per source for a quick test (0 = no limit).
- `--name-thresh` / `--addr-thresh` _(floats)_: optional fuzzy thresholds if enabled in your version.
- `--address-first` _(flag)_: prefer address-based grouping when available.

## Normalization (high level)

- Lowercasing, German umlaut transliteration, punctuation/whitespace squashing.
- Common legal tokens removed: `gmbh`, `ggmbh`, `mbh`, `ev/e.v.`, `verein`, etc.
- AWO synonyms: `arbeiterwohlfahrt` → `awo`.
- Abbrev expansion: `ov` ↔︎ `ortsverein`, `bv` ↔︎ `bezirksverband`, `kv` ↔︎ `kreisverband`, `lv` ↔︎ `landesverband`.
- Address normalization: house numbers split, street/city/ZIP cleaned.

> The matching key is primarily **normalized name + normalized address** per source; no AWO numbers are used.

## Privacy

Outputs may include _single values_ (unique names, addresses). Share only within approved groups and follow your data-sharing rules.
