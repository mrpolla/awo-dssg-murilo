#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geocoding script for AWO entities.
Reads entities.csv and addresses_unique.csv, geocodes addresses,
and outputs a complete geocoded CSV file.
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from pathlib import Path
import pickle
import sys
from datetime import datetime

# ==================== CONFIG ====================
ENTITIES_FILE = Path("data/output/entities.csv")
ADDRESSES_FILE = Path("data/output/addresses_unique.csv")
OUTPUT_FILE = Path("data/output/awo_entities_geocoded_complete.csv")
GEOCODE_CACHE_FILE = Path("data/output/geocode_cache.pkl")
FAILED_LOG_FILE = Path("data/output/failed_geocoding.csv")

SAMPLE_LIMIT = 1000  # Set to a number for testing, or None for all data

# ================================================

def fix_zip_code(zip_val):
    """Fix ZIP code formatting - ensure 5 digits, pad with 0 if needed."""
    if pd.isna(zip_val):
        return None
    
    # Convert to string and remove .0 if present
    zip_str = str(zip_val).strip()
    if zip_str.endswith('.0'):
        zip_str = zip_str[:-2]
    
    # Remove any non-digit characters
    zip_str = ''.join(c for c in zip_str if c.isdigit())
    
    if not zip_str:
        return None
    
    # Pad with leading zeros to make 5 digits
    zip_str = zip_str.zfill(5)
    
    # German ZIP codes are exactly 5 digits
    if len(zip_str) == 5:
        return zip_str
    elif len(zip_str) > 5:
        # Take first 5 digits if too long
        return zip_str[:5]
    else:
        return None

def load_entities():
    """Load the entities data from CSV."""
    print("Loading entities...")
    df = pd.read_csv(ENTITIES_FILE)
    
    # Ensure boolean flags
    flag_cols = ["IsFacility", "IsAssociation", "IsLegalEntity", "IsAWODomain"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    
    print(f"✓ Loaded {len(df)} entities")
    return df

def load_unique_addresses():
    """Load unique addresses from addresses_unique.csv."""
    print("Loading addresses...")
    df = pd.read_csv(ADDRESSES_FILE)
    
    # Apply sample limit if set
    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(df):
        df = df.sample(n=SAMPLE_LIMIT, random_state=42)
        print(f"  Using sample of {SAMPLE_LIMIT} addresses")
    
    addresses = []
    skipped = 0
    
    for idx, row in df.iterrows():
        # Fix ZIP code
        zip_code = fix_zip_code(row.get("ZIP"))
        if not zip_code:
            skipped += 1
            continue
        
        city = str(row.get("City_norm", "")).strip()
        street = str(row.get("Street_norm", "")).strip()
        
        if not city or city == "nan":
            skipped += 1
            continue
        
        # Parse EntityIDs (comma-separated)
        entity_ids_str = str(row.get("EntityIDs", "")).strip()
        if entity_ids_str and entity_ids_str != "nan":
            n_entities = len([x for x in entity_ids_str.split(",") if x.strip()])
        else:
            entity_ids_str = ""
            n_entities = 0
        
        addr_key = f"{zip_code}|{city}|{street}" if street and street != "nan" else f"{zip_code}|{city}|"
        full_address = f"{street}, {zip_code} {city}" if street and street != "nan" else f"{zip_code} {city}"
        
        addresses.append({
            "zip": zip_code,
            "city": city,
            "street": street if street != "nan" else "",
            "addr_key": addr_key,
            "full_address": full_address,
            "entity_ids_str": entity_ids_str,
            "n_entities": n_entities
        })
    
    print(f"✓ Loaded {len(addresses)} valid addresses (skipped {skipped})")
    return pd.DataFrame(addresses)

def load_geocode_cache():
    """Load the persistent geocode cache from disk."""
    if GEOCODE_CACHE_FILE.exists():
        try:
            with open(GEOCODE_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                print(f"✓ Loaded {len(cache)} cached geocodes")
                return cache
        except Exception as e:
            print(f"⚠ Could not load cache: {e}")
            return {}
    return {}

def save_geocode_cache(cache):
    """Save the geocode cache to disk."""
    try:
        GEOCODE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GEOCODE_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        return True
    except Exception as e:
        print(f"⚠ Could not save cache: {e}")
        return False

def geocode_addresses(addresses_df):
    """Geocode unique addresses to get lat/lon coordinates."""
    print("\n" + "="*60)
    print("GEOCODING ADDRESSES")
    print("="*60)
    
    # Load existing cache
    cache = load_geocode_cache()
    
    # Initialize failed addresses log
    failed_log = {}
    
    # Filter out already geocoded addresses
    to_geocode = addresses_df[~addresses_df["addr_key"].isin(cache.keys())]
    
    if len(to_geocode) == 0:
        print("✓ All addresses already geocoded!")
        coords = [{"addr_key": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()]
        return pd.DataFrame(coords), failed_log
    
    print(f"\nNeed to geocode {len(to_geocode)} new addresses")
    print(f"Estimated time: ~{len(to_geocode)} seconds (1 per second with rate limiting)")
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="awo_entity_mapper", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    success_count = 0
    fail_count = 0
    
    print("\nStarting geocoding...")
    print("-" * 60)
    
    for idx, (_, row) in enumerate(to_geocode.iterrows()):
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(to_geocode)} | Success: {success_count} | Failed: {fail_count}")
        
        try:
            location = geocode(row["full_address"] + ", Germany")
            if location:
                cache[row["addr_key"]] = (location.latitude, location.longitude)
                success_count += 1
            else:
                # Try without street if full address fails
                simple_addr = f"{row['zip']} {row['city']}, Germany"
                location = geocode(simple_addr)
                if location:
                    cache[row["addr_key"]] = (location.latitude, location.longitude)
                    success_count += 1
                else:
                    failed_log[row["addr_key"]] = {"address": row["full_address"], "reason": "Not found"}
                    fail_count += 1
        except Exception as e:
            failed_log[row["addr_key"]] = {"address": row["full_address"], "reason": str(e)}
            fail_count += 1
            continue
        
        # Save cache every 50 addresses
        if success_count > 0 and success_count % 50 == 0:
            save_geocode_cache(cache)
    
    print("-" * 60)
    print(f"✓ Geocoding complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total in cache: {len(cache)}")
    
    # Save final cache
    save_geocode_cache(cache)
    
    # Return all coords including cached ones
    all_coords = [{"addr_key": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()]
    return pd.DataFrame(all_coords), failed_log

def save_failed_log(failed_log):
    """Save failed geocoding attempts to CSV."""
    if not failed_log:
        return
    
    failed_df = pd.DataFrame([
        {"Address": v["address"], "Reason": v["reason"]} 
        for k, v in failed_log.items()
    ])
    
    try:
        FAILED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        failed_df.to_csv(FAILED_LOG_FILE, index=False)
        print(f"✓ Saved failed addresses to {FAILED_LOG_FILE}")
    except Exception as e:
        print(f"⚠ Could not save failed log: {e}")

def main():
    """Main geocoding workflow."""
    print("\n" + "="*60)
    print("AWO ENTITIES GEOCODING SCRIPT")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if input files exist
    if not ENTITIES_FILE.exists():
        print(f"❌ Error: {ENTITIES_FILE} not found!")
        sys.exit(1)
    
    if not ADDRESSES_FILE.exists():
        print(f"❌ Error: {ADDRESSES_FILE} not found!")
        sys.exit(1)
    
    # Load data
    df_entities = load_entities()
    addresses = load_unique_addresses()
    
    if len(addresses) == 0:
        print("❌ Error: No valid addresses found!")
        sys.exit(1)
    
    # Geocode addresses
    coords, failed_log = geocode_addresses(addresses)
    
    if len(coords) == 0:
        print("❌ Error: No coordinates found!")
        sys.exit(1)
    
    # Save failed log
    if failed_log:
        save_failed_log(failed_log)
    
    # Merge coordinates with addresses
    print("\nMerging data...")
    data = addresses.merge(coords, on="addr_key", how="inner")
    
    # Expand data: one row per entity
    print("Expanding entities...")
    data["entity_ids"] = data["entity_ids_str"].apply(
        lambda x: [int(eid.strip()) for eid in str(x).split(",") if eid.strip()] if x else []
    )
    data_expanded = data.explode("entity_ids").rename(columns={"entity_ids": "EntityID"})
    data_expanded = data_expanded[data_expanded["EntityID"].notna()]
    data_expanded["EntityID"] = data_expanded["EntityID"].astype(int)
    
    # Join with entity data
    print("Joining with entity details...")
    data_expanded = data_expanded.merge(
        df_entities[["EntityID", "EntityName", "IsFacility", "IsAssociation", "IsLegalEntity", "IsAWODomain"]], 
        on="EntityID", 
        how="left"
    )
    
    # Save output
    print(f"\nSaving output to {OUTPUT_FILE}...")
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        data_expanded.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Saved {len(data_expanded)} entity records to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving output: {e}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total entities: {len(data_expanded)}")
    print(f"Unique addresses: {len(data_expanded[['lat', 'lon', 'addr_key']].drop_duplicates())}")
    print(f"Successfully geocoded: {len(coords)}")
    print(f"Failed to geocode: {len(failed_log)}")
    print(f"\nOutput file: {OUTPUT_FILE}")
    if failed_log:
        print(f"Failed log: {FAILED_LOG_FILE}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()