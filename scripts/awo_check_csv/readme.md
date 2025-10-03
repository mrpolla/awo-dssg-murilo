# AWO Website Checker & Address Scraper

A Python script that validates AWO organization websites and extracts contact information (addresses and emails) from their web pages.

**⚠️ Note: This script is currently inefficient and slow.** It tries multiple extraction strategies and visits many pages per website. Performance optimizations are planned for future versions.

## What It Does

- Checks if websites are reachable
- Extracts German addresses using multi-stage pattern matching
- Finds email addresses on main pages and contact/impressum pages
- Compares found data with existing records and provides similarity scores

## Requirements

```bash
pip install requests beautifulsoup4 rapidfuzz lxml
# or
pip install -r requirements.txt
```

## Input CSV Format

Required columns: `id`, `domain`, `name`, `strasse`, `hausnummer`, `plz`, `ort`, `email`

## Usage

### Basic (uses default paths)

```bash
python /scripts/awo_check_csv.py
```

### Custom input/output

```bash
python /scripts/awo_check_csv.py --input /data/input/your_file.csv --output results.csv
```

### From VS Code

Run directly in VS Code by manually changing the default input file path in the script or using the integrated terminal with the above commands.

### Options

- `--input`: Input CSV path (default: `/data/input/demo_seed.csv`)
- `--output`: Output CSV path (default: `/data/output/scraped_results.csv`)
- `--max N`: Process only first N rows (for testing)
- `--delay X`: Seconds between requests (default: 1.0)

## Output

Generates CSV with original data plus found addresses, emails, similarity scores, and contact page URLs.
