# AWO Entity Matching & Utilities

This repository contains two scripts:

- **consolidate_lists** — merges Facilities, Associations, Legal Entities and AWO Domains into a single, deduplicated _Master Entities_ list (one row per entity), with overlap/summary sheets.
- **awo_check_csv** — checks addresses and domains from a seed list of AWO domains.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### Repo layout

```
awo-dssg-murilo/
├─ README.md
├─ scripts/
│  ├─ consolidate_lists/
│  │  ├─ README.md
│  │  └─ consolidate_lists.py
│  └─ awo_check_csv/
│     ├─ README.md
│     └─ awo_check_csv.py
├─ data/
│  ├─ input/    # place source files here (not committed)
│  └─ output/   # results (not committed)
└─ .gitignore
```

**Privacy note (Datenschutz):** the outputs may include _single values_ from the sources (e.g., unique names/addresses). Do not publish raw outputs without checking data-sharing permissions.
