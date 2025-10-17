#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app to visualize AWO entities on a map of Germany.
Loads pre-geocoded data and displays with interactive filters.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path

# ==================== CONFIG ====================
DEFAULT_GEOCODED_FILE = Path("data/output/awo_entities_geocoded_complete.csv")

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
def load_geocoded_data(file_path=None, uploaded_file=None):
    """Load pre-geocoded data from CSV file."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif file_path is not None:
            df = pd.read_csv(file_path)
        else:
            return None
        
        # Ensure boolean columns are proper booleans
        for col in ["IsFacility", "IsAssociation", "IsLegalEntity", "IsAWODomain"]:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        
        # Verify required columns exist
        required_cols = ["lat", "lon", "addr_key", "EntityID", "EntityName", "full_address"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

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

def show_map(data_expanded):
    """Display the map with filters and entity details."""
    
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
        
        # Create dict of active filters for dynamic coloring
        active_filters = {
            "Facility": show_facility,
            "Association": show_association,
            "LegalEntity": show_legal,
            "AWODomain": show_domain
        }
        
        # Group entities by location
        location_groups = filtered.groupby(["lat", "lon", "addr_key"])
        
        for (lat, lon, addr_key), group in location_groups:
            entity_types = []
            for _, row in group.iterrows():
                entity_types.extend(get_entity_types(row))
            entity_types = list(set(entity_types))
            
            # FILTER entity types to only include active/checked ones
            entity_types_active = [t for t in entity_types if active_filters.get(t, False)]
            
            # Skip if no active types (shouldn't happen due to filtering, but safety check)
            if not entity_types_active:
                continue
            
            # Create popup content
            popup_html = f"<b>{group.iloc[0]['full_address']}</b><br>"
            popup_html += f"<b>{len(group)} entities</b><br><br>"
            
            for _, row in group.iterrows():
                types = get_entity_types(row)
                type_str = ", ".join(types)
                popup_html += f"• {row['EntityName']}<br>"
                popup_html += f"  <i>({type_str})</i><br>"
            
            # Get color based on ACTIVE types only
            color = get_primary_color(entity_types_active)
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

def main():
    st.title("🗺️ AWO Entities Map Viewer")
    
    # Initialize session state
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.data = None
    
    # If data not loaded yet, show load options
    if not st.session_state.data_loaded:
        st.markdown("### Load geocoded data:")
        
        # Check if default file exists
        default_exists = DEFAULT_GEOCODED_FILE.exists()
        
        if default_exists:
            st.info(f"📂 Found default file: `{DEFAULT_GEOCODED_FILE}`")
            if st.button("📥 Load Default File", type="primary"):
                with st.spinner("Loading..."):
                    data = load_geocoded_data(file_path=DEFAULT_GEOCODED_FILE)
                    if data is not None:
                        st.success(f"✓ Loaded {len(data)} entities from {len(data[['lat', 'lon', 'addr_key']].drop_duplicates())} unique addresses")
                        st.session_state.data = data
                        st.session_state.data_loaded = True
                        st.rerun()
        
        st.markdown("**Or upload your own file:**")
        uploaded_file = st.file_uploader(
            "Upload geocoded CSV file",
            type=["csv"],
            help="Upload a complete geocoded CSV file (e.g., awo_entities_geocoded_complete.csv)"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading file..."):
                data = load_geocoded_data(uploaded_file=uploaded_file)
                if data is not None:
                    st.success(f"✓ Loaded {len(data)} entities from {len(data[['lat', 'lon', 'addr_key']].drop_duplicates())} unique addresses")
                    st.session_state.data = data
                    st.session_state.data_loaded = True
                    st.rerun()
        
        if not default_exists:
            st.warning("⚠️ No default geocoded file found. Please upload a file or run `geocode_addresses.py` first.")
    
    # If data is loaded, show the map
    else:
        # Add reset button in sidebar
        if st.sidebar.button("🔄 Load Different File"):
            st.session_state.data_loaded = False
            st.session_state.data = None
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Show the map
        show_map(st.session_state.data)

if __name__ == "__main__":
    main()