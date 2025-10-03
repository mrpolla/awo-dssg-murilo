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

## Run

- From within VS Code

### Key options

- MAX_PER_SOURCE: limit rows per source for a quick test (None = no limit).

## Normalization

- Lowercasing, German umlaut transliteration, punctuation/whitespace squashing.
- Common legal tokens removed: `gmbh`, `ggmbh`, `mbh`, `ev/e.v.`, `verein`, etc.
- AWO synonyms: `arbeiterwohlfahrt` → `awo`.
- Abbrev expansion: `ov` ↔︎ `ortsverein`, `bv` ↔︎ `bezirksverband`, `kv` ↔︎ `kreisverband`, `lv` ↔︎ `landesverband`.
- Address normalization: house numbers split, street/city/ZIP cleaned.

> The matching key is primarily **normalized name + normalized address** per source; no AWO numbers are used.

## Privacy

**Privacy note (Datenschutz):** the outputs may include _single values_ from the sources (e.g., unique names/addresses). Do not publish raw outputs without checking data-sharing permissions.
