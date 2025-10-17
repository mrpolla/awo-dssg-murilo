#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app to visualize AWO entities on a map of Germany.
Filters by entity type and shows details when clicking on locations.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
from pathlib import Path
import pickle

# ==================== CONFIG ====================
DATA_FILE = Path("data/output/deduplicated_entities.xlsx")
GEOCODE_CACHE_FILE = Path("../../data/output/geocode_cache.pkl")
SAMPLE_LIMIT = 100  # Set to a number for testing, or None for all data

# ================================================

# Page config
st.set_page_config(page_title="AWO Entities Map", layout="wide")

# Color scheme for entity types
COLORS = {
    "Facility": "#e74c3c",      # red
    "Association": "#3498db",    # blue
    "LegalEntity": "#2ecc71",    # green
    "AWODomain": "#f39c12"       # orange
}

@st.cache_data
def load_data():
    """Load the entities data from Excel."""
    df = pd.read_excel(DATA_FILE, sheet_name="Entities")
    
    # Ensure boolean flags
    flag_cols = ["IsFacility", "IsAssociation", "IsLegalEntity", "IsAWODomain"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    
    # No sample limit here - we load all entities and filter by addresses
    return df

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

@st.cache_data
def load_unique_addresses():
    """Load unique addresses from the Addr_Name_Collisions sheet."""
    df = pd.read_excel(DATA_FILE, sheet_name="Addr_Name_Collisions")
    
    # Apply sample limit to addresses if set
    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(df):
        df = df.sample(n=SAMPLE_LIMIT, random_state=42)
    
    addresses = []
    for idx, row in df.iterrows():
        # Fix ZIP code
        zip_code = fix_zip_code(row.get("ZIP"))
        if not zip_code:
            continue
        
        city = str(row.get("City_norm", "")).strip()
        street = str(row.get("Street_norm", "")).strip()
        
        if not city or city == "nan":
            continue
        
        # Parse EntityIDs (comma-separated) - keep as STRING for caching
        entity_ids_str = str(row.get("EntityIDs", "")).strip()
        if entity_ids_str and entity_ids_str != "nan":
            # Store as comma-separated string, not list
            entity_ids = entity_ids_str
            n_entities = len([x for x in entity_ids_str.split(",") if x.strip()])
        else:
            entity_ids = ""
            n_entities = 0
        
        addr_key = f"{zip_code}|{city}|{street}" if street and street != "nan" else f"{zip_code}|{city}|"
        full_address = f"{street}, {zip_code} {city}" if street and street != "nan" else f"{zip_code} {city}"
        
        addresses.append({
            "zip": zip_code,
            "city": city,
            "street": street if street != "nan" else "",
            "addr_key": addr_key,
            "full_address": full_address,
            "entity_ids_str": entity_ids,  # String, not list
            "n_entities": n_entities
        })
    
    return pd.DataFrame(addresses)

def load_geocode_cache():
    """Load the persistent geocode cache from disk."""
    if GEOCODE_CACHE_FILE.exists():
        try:
            with open(GEOCODE_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_geocode_cache(cache):
    """Save the geocode cache to disk."""
    try:
        GEOCODE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GEOCODE_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        return True
    except Exception:
        return False

@st.cache_data
def geocode_addresses(addresses_df):
    """Geocode unique addresses to get lat/lon coordinates with persistent caching."""
    # Only use columns needed for geocoding (avoid unhashable lists)
    geocode_df = addresses_df[["addr_key", "full_address", "zip", "city"]].copy()
    
    # Load existing cache
    cache = load_geocode_cache()
    
    # Load or initialize failed addresses log
    failed_log = {}
    
    # Filter out already geocoded addresses
    to_geocode = geocode_df[~geocode_df["addr_key"].isin(cache.keys())]
    
    if len(to_geocode) == 0:
        coords = [{"addr_key": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()]
        return pd.DataFrame(coords), failed_log
    
    # Initialize geocoder with longer timeout
    geolocator = Nominatim(user_agent="awo_entity_mapper", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Geocoding {len(to_geocode)} new addresses...")
    
    success_count = 0
    fail_count = 0
    
    for idx, (_, row) in enumerate(to_geocode.iterrows()):
        progress_bar.progress((idx + 1) / len(to_geocode))
        
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
    
    progress_bar.empty()
    status_text.empty()
    
    # Show summary
    if success_count > 0 or fail_count > 0:
        st.info(f"✓ Geocoded {success_count} | ✗ Failed {fail_count}")
    
    # Save final cache
    save_geocode_cache(cache)
    
    # Return all coords including cached ones
    all_coords = [{"addr_key": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()]
    return pd.DataFrame(all_coords), failed_log

def get_entity_types(row):
    """Get list of entity types for a given entity."""
    types = []
    if row.get("IsFacility", False):
        types.append("Facility")
    if row.get("IsAssociation", False):
        types.append("Association")
    if row.get("IsLegalEntity", False):
        types.append("LegalEntity")
    if row.get("IsAWODomain", False):
        types.append("AWODomain")
    return types

def get_primary_color(types):
    """Get the primary color for marker (first type in priority order)."""
    priority = ["Association", "LegalEntity", "Facility", "AWODomain"]
    for t in priority:
        if t in types:
            return COLORS[t]
    return "#95a5a6"  # gray fallback

def main():
    st.title("🗺️ AWO Entities Map Viewer")
    
    # Load entity data
    df_entities = load_data()
    
    # Load unique addresses from Addr_Name_Collisions sheet
    addresses = load_unique_addresses()
    
    if len(addresses) == 0:
        st.warning("No valid addresses found in Addr_Name_Collisions sheet.")
        return
    
    # Geocode unique addresses (cached, runs once)
    coords, failed_log = geocode_addresses(addresses)
    
    # EXPORT SECTION - at the top for visibility
    st.success(f"✓ Geocoded {len(coords)} / {len(addresses)} unique addresses")
    
    col_export1, col_export2, col_export3 = st.columns([1, 1, 1])
    
    with col_export1:
        if len(coords) > 0:
            export_data = addresses.merge(coords, on="addr_key", how="inner")
            # entity_ids_str is already a string, ready for export
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Geocoded Data",
                data=csv,
                file_name="geocoded_addresses.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_export2:
        if len(failed_log) > 0:
            failed_df = pd.DataFrame([
                {"Address": v["address"], "Reason": v["reason"]} 
                for k, v in failed_log.items()
            ])
            failed_csv = failed_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Failed Addresses ({len(failed_log)})",
                data=failed_csv,
                file_name="failed_geocoding.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_export3:
        if len(failed_log) > 0:
            with st.expander("View Failed Addresses"):
                failed_df = pd.DataFrame([
                    {"Address": v["address"], "Reason": v["reason"]} 
                    for k, v in failed_log.items()
                ])
                st.dataframe(failed_df, use_container_width=True)
    
    st.markdown("---")
    
    # SIDEBAR - filters and legend
    st.sidebar.header("🔍 Filter by Type")
    show_facility = st.sidebar.checkbox("🏢 Facilities", value=True)
    show_association = st.sidebar.checkbox("🤝 Associations", value=True)
    show_legal = st.sidebar.checkbox("⚖️ Legal Entities", value=True)
    show_domain = st.sidebar.checkbox("🌐 AWO Domains", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Legend")
    for entity_type, color in COLORS.items():
        st.sidebar.markdown(
            f"<span style='color:{color}'>●</span> {entity_type}",
            unsafe_allow_html=True
        )
    
    if len(coords) == 0:
        st.warning("No coordinates found for addresses.")
        return
    
    # Merge coordinates with addresses
    data = addresses.merge(coords, on="addr_key", how="inner")
    
    if len(data) == 0:
        st.warning("No coordinates found for addresses.")
        return
    
    # Expand data: one row per entity (explode the entity_ids list)
    # First convert entity_ids_str back to list of integers
    data["entity_ids"] = data["entity_ids_str"].apply(
        lambda x: [int(eid.strip()) for eid in str(x).split(",") if eid.strip()] if x else []
    )
    data_expanded = data.explode("entity_ids").rename(columns={"entity_ids": "EntityID"})
    data_expanded = data_expanded[data_expanded["EntityID"].notna()]
    data_expanded["EntityID"] = data_expanded["EntityID"].astype(int)
    
    # Join with entity data to get entity details and type flags
    data_expanded = data_expanded.merge(
        df_entities[["EntityID", "EntityName", "IsFacility", "IsAssociation", "IsLegalEntity", "IsAWODomain"]], 
        on="EntityID", 
        how="left"
    )
    
    # Apply filters
    filtered = data_expanded.copy()
    type_filters = []
    if show_facility:
        type_filters.append(filtered["IsFacility"])
    if show_association:
        type_filters.append(filtered["IsAssociation"])
    if show_legal:
        type_filters.append(filtered["IsLegalEntity"])
    if show_domain:
        type_filters.append(filtered["IsAWODomain"])
    
    if type_filters:
        mask = type_filters[0]
        for f in type_filters[1:]:
            mask = mask | f
        filtered = filtered[mask]
    else:
        filtered = pd.DataFrame()
    
    st.sidebar.metric("Visible Entities", len(filtered))
    
    if len(filtered) == 0:
        st.info("No entities match the selected filters.")
        return
    
    # Create map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Map View")
        
        # Center map on Germany
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles="OpenStreetMap")
        
        # Group entities by location
        location_groups = filtered.groupby(["lat", "lon", "addr_key"])
        
        for (lat, lon, addr_key), group in location_groups:
            entity_types = []
            for _, row in group.iterrows():
                entity_types.extend(get_entity_types(row))
            entity_types = list(set(entity_types))
            
            # Create popup content
            popup_html = f"<b>{group.iloc[0]['full_address']}</b><br>"
            popup_html += f"<b>{len(group)} entities</b><br><br>"
            
            for _, row in group.iterrows():
                types = get_entity_types(row)
                type_str = ", ".join(types)
                popup_html += f"• {row['EntityName']}<br>"
                popup_html += f"  <i>({type_str})</i><br>"
            
            # Add marker with primary color
            color = get_primary_color(entity_types)
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=800, height=600)
    
    with col2:
        st.subheader("Entity Details")
        
        # Show selected location details
        if map_data and map_data.get("last_object_clicked"):
            clicked = map_data["last_object_clicked"]
            clicked_lat = clicked["lat"]
            clicked_lon = clicked["lng"]
            
            # Find entities at this location
            location_entities = filtered[
                (filtered["lat"].round(4) == round(clicked_lat, 4)) & 
                (filtered["lon"].round(4) == round(clicked_lon, 4))
            ]
            
            if len(location_entities) > 0:
                st.write(f"**Address:** {location_entities.iloc[0]['full_address']}")
                st.write(f"**{len(location_entities)} entities at this location:**")
                st.write("---")
                
                for _, entity in location_entities.iterrows():
                    types = get_entity_types(entity)
                    type_badges = " ".join([f"`{t}`" for t in types])
                    
                    st.markdown(f"**{entity['EntityName']}**")
                    st.markdown(type_badges)
                    st.markdown(f"*EntityID: {entity['EntityID']}*")
                    st.write("")
        else:
            st.info("👈 Click on a marker on the map to see entity details")

if __name__ == "__main__":
    main()