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
SAMPLE_LIMIT = 1000  # Set to a number for testing, or None for all data

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
    
    # Apply sample limit if set
    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(df):
        df = df.sample(n=SAMPLE_LIMIT, random_state=42)
    
    return df

@st.cache_data
def extract_addresses(df):
    """Extract unique addresses from the entity data."""
    addresses = []
    
    for idx, row in df.iterrows():
        zip_code = ""
        city = ""
        street = ""
        
        # Extract from available source columns
        for prefix in ["F_", "A_", "L_"]:
            if not zip_code:
                plz_col = f"{prefix}adresse_plz" if prefix != "L_" else "L_PLZ"
                if plz_col in df.columns:
                    val = str(row.get(plz_col, "")).strip()
                    if val and val != "nan":
                        zip_code = val
            
            if not city:
                city_col = f"{prefix}adresse_ort" if prefix != "L_" else "L_Ort"
                if city_col in df.columns:
                    val = str(row.get(city_col, "")).strip()
                    if val and val != "nan":
                        city = val
            
            if not street:
                street_col = f"{prefix}adresse_strasse" if prefix != "L_" else "L_Straße + Hausnr."
                if street_col in df.columns:
                    val = str(row.get(street_col, "")).strip()
                    if val and val != "nan":
                        street = val
        
        # Try domain columns
        if not zip_code and "D_plz" in df.columns:
            val = str(row.get("D_plz", "")).strip()
            if val and val != "nan":
                zip_code = val
        if not city and "D_ort" in df.columns:
            val = str(row.get("D_ort", "")).strip()
            if val and val != "nan":
                city = val
        if not street and "D_strasse" in df.columns:
            street_val = str(row.get("D_strasse", "")).strip()
            house_val = str(row.get("D_hausnummer", "")).strip()
            if street_val and street_val != "nan":
                street = f"{street_val} {house_val}".strip()
        
        if zip_code and city:
            addr_key = f"{zip_code}|{city}|{street}"
            addresses.append({
                "EntityID": row["EntityID"],
                "zip": zip_code,
                "city": city,
                "street": street,
                "addr_key": addr_key,
                "full_address": f"{street}, {zip_code} {city}" if street else f"{zip_code} {city}"
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
    unique_addrs = addresses_df[["addr_key", "full_address"]].drop_duplicates()
    
    # Load existing cache
    cache = load_geocode_cache()
    
    # Load or initialize failed addresses log
    failed_log = {}
    
    # Filter out already geocoded addresses
    to_geocode = unique_addrs[~unique_addrs["addr_key"].isin(cache.keys())]
    
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
                simple_addr = f"{row['addr_key'].split('|')[0]} {row['addr_key'].split('|')[1]}, Germany"
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
    
    # SIDEBAR - only filters and legend
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
    
    # Load ALL data once (not filtered)
    df_entities = load_data()
    
    # Extract and geocode ALL addresses once (cached)
    addresses = extract_addresses(df_entities)
    
    if len(addresses) == 0:
        st.warning("No valid addresses found in entities.")
        return
    
    # Geocode with persistent caching (cached, runs once)
    coords, failed_log = geocode_addresses(addresses)
    
    if len(coords) == 0:
        st.warning("No coordinates found for entities.")
        return
    
    # Merge coordinates with addresses and entities
    data = addresses.merge(coords, on="addr_key", how="inner")
    data = data.merge(df_entities[["EntityID", "EntityName", "IsFacility", "IsAssociation", 
                                 "IsLegalEntity", "IsAWODomain"]], 
                      on="EntityID", how="left")
    
    # NOW apply filters for display only
    filtered = data.copy()
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