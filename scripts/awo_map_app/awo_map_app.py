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

# Page config
st.set_page_config(page_title="AWO Entities Map", layout="wide")

# File path
DATA_FILE = Path("data/output/deduplicated_entities.xlsx")

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
    
    return df

@st.cache_data
def extract_addresses(df):
    """Extract unique addresses from the entity data."""
    addresses = []
    
    for idx, row in df.iterrows():
        # Try to extract address from different source columns
        # Priority: Facility > Association > Legal > Domain
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

@st.cache_data
def geocode_addresses(addresses_df):
    """Geocode unique addresses to get lat/lon coordinates."""
    unique_addrs = addresses_df[["addr_key", "full_address"]].drop_duplicates()
    
    geolocator = Nominatim(user_agent="awo_entity_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    coords = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in unique_addrs.iterrows():
        status_text.text(f"Geocoding {idx + 1}/{len(unique_addrs)}: {row['full_address']}")
        progress_bar.progress((idx + 1) / len(unique_addrs))
        
        try:
            location = geocode(row["full_address"] + ", Germany")
            if location:
                coords.append({
                    "addr_key": row["addr_key"],
                    "lat": location.latitude,
                    "lon": location.longitude
                })
            else:
                # Try without street if full address fails
                simple_addr = f"{row['addr_key'].split('|')[0]} {row['addr_key'].split('|')[1]}, Germany"
                location = geocode(simple_addr)
                if location:
                    coords.append({
                        "addr_key": row["addr_key"],
                        "lat": location.latitude,
                        "lon": location.longitude
                    })
        except Exception as e:
            st.warning(f"Could not geocode: {row['full_address']}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(coords)

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
    
    # Load data
    with st.spinner("Loading entity data..."):
        df_entities = load_data()
    
    st.sidebar.header("Filters")
    
    # Entity type filters
    st.sidebar.subheader("Entity Types")
    show_facility = st.sidebar.checkbox("🏢 Facilities", value=True, 
                                        help=f"Color: {COLORS['Facility']}")
    show_association = st.sidebar.checkbox("🤝 Associations", value=True,
                                           help=f"Color: {COLORS['Association']}")
    show_legal = st.sidebar.checkbox("⚖️ Legal Entities", value=True,
                                     help=f"Color: {COLORS['LegalEntity']}")
    show_domain = st.sidebar.checkbox("🌐 AWO Domains", value=True,
                                      help=f"Color: {COLORS['AWODomain']}")
    
    # Filter entities
    filtered = df_entities.copy()
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
    
    st.sidebar.metric("Filtered Entities", len(filtered))
    
    if len(filtered) == 0:
        st.warning("No entities match the selected filters.")
        return
    
    # Extract and geocode addresses
    with st.spinner("Processing addresses..."):
        addresses = extract_addresses(filtered)
    
    if len(addresses) == 0:
        st.warning("No valid addresses found in filtered entities.")
        return
    
    # Check if we need to geocode
    if "coords_cache" not in st.session_state:
        with st.spinner("Geocoding addresses (this may take a few minutes)..."):
            coords = geocode_addresses(addresses)
            st.session_state.coords_cache = coords
    else:
        coords = st.session_state.coords_cache
    
    # Merge coordinates with addresses and entities
    data = addresses.merge(coords, on="addr_key", how="inner")
    data = data.merge(filtered[["EntityID", "EntityName", "IsFacility", "IsAssociation", 
                                 "IsLegalEntity", "IsAWODomain"]], 
                      on="EntityID", how="left")
    
    if len(data) == 0:
        st.warning("No coordinates found for filtered entities.")
        return
    
    # Create map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Map View")
        
        # Center map on Germany
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles="OpenStreetMap")
        
        # Group entities by location
        location_groups = data.groupby(["lat", "lon", "addr_key"])
        
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
            location_entities = data[
                (data["lat"].round(4) == round(clicked_lat, 4)) & 
                (data["lon"].round(4) == round(clicked_lon, 4))
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
    
    # Legend
    st.sidebar.markdown("---")
    st.sidebar.subheader("Legend")
    for entity_type, color in COLORS.items():
        st.sidebar.markdown(
            f"<span style='color:{color}'>●</span> {entity_type}",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()