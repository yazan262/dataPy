from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

st.set_page_config(page_title="Wolken-Effekt Analyse", layout="wide")
st.title("☁️ Wolken-Effekt: Aufenthaltsorte je nach Wetter")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "zebra_weather_cleaned.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_data(DATA_PATH)

# Sidebar Filter
st.sidebar.header("Filter & Einstellungen")

# Year filter
years = sorted(df["timestamp"].dt.year.dropna().unique())
if "wolken_year" not in st.session_state:
    st.session_state.wolken_year = years[0] if years else None

year = st.sidebar.selectbox(
    "Jahr",
    years,
    index=(years.index(st.session_state.wolken_year) if st.session_state.wolken_year in years else 0),
    key="wolken_year_select",
)
st.session_state.wolken_year = year

# Zebra filter
df_year = df[df["timestamp"].dt.year == year]
zebra_ids_year = sorted(df_year["individual_local_identifier"].dropna().unique())

if "wolken_zebra_ids" not in st.session_state:
    st.session_state.wolken_zebra_ids = []

zebra_ids_selected = st.sidebar.multiselect(
    "Zebra-IDs (aus gewähltem Jahr)",
    zebra_ids_year,
    default=st.session_state.wolken_zebra_ids,
    key="wolken_zebra_select",
)
st.session_state.wolken_zebra_ids = zebra_ids_selected

# Temperature threshold slider for Visualization A
st.sidebar.divider()
st.sidebar.subheader("Temperatur-Schwellenwerte")
temp_threshold_cold = st.sidebar.slider(
    "Kalt-Schwellenwert (°C)",
    min_value=-10.0,
    max_value=30.0,
    value=15.0,
    step=0.5,
    help="Datenpunkte mit T2M < diesem Wert werden als 'Kalt' klassifiziert",
)

temp_threshold_hot = st.sidebar.slider(
    "Heiß-Schwellenwert (°C)",
    min_value=10.0,
    max_value=50.0,
    value=30.0,
    step=0.5,
    help="Datenpunkte mit T2M > diesem Wert werden als 'Heiß' klassifiziert",
)

# Filter data
if not year:
    st.error("Keine Jahre im Datensatz gefunden.")
    st.stop()

if not zebra_ids_selected:
    st.warning("Bitte wähle mindestens eine Zebra-ID aus.")
    st.stop()

df_filtered = df[
    (df["timestamp"].dt.year == year)
    & (df["individual_local_identifier"].isin(zebra_ids_selected))
].copy()

df_filtered = df_filtered.dropna(subset=["location_lat", "location_long", "T2M"])

if df_filtered.empty:
    st.error("Keine Daten für die Auswahl vorhanden.")
    st.stop()

# Info metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("GPS-Punkte", f"{len(df_filtered):,}")
col2.metric("Ø Temperatur", f"{df_filtered['T2M'].mean():.1f}°C")
col3.metric("Min Temperatur", f"{df_filtered['T2M'].min():.1f}°C")
col4.metric("Max Temperatur", f"{df_filtered['T2M'].max():.1f}°C")

st.divider()

# Tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["☁️ KDE Plot (Density)", "📍 Scatter Map", "⬡ Hexbin Plot"])

# ============================================================================
# TAB 1: KDE / Density Plot
# ============================================================================
with tab1:
    st.subheader("KDE/Density Plot: Kalt vs. Heiß")
    st.caption(
        f"Linker Plot: Datenpunkte mit T2M < {temp_threshold_cold}°C (Kalt) | "
        f"Rechter Plot: Datenpunkte mit T2M > {temp_threshold_hot}°C (Heiß)"
    )

    # Filter data for cold and hot
    df_cold = df_filtered[df_filtered["T2M"] < temp_threshold_cold].copy()
    df_hot = df_filtered[df_filtered["T2M"] > temp_threshold_hot].copy()

    if df_cold.empty and df_hot.empty:
        st.warning(
            f"Keine Datenpunkte gefunden für die gewählten Schwellenwerte "
            f"(Kalt: < {temp_threshold_cold}°C, Heiß: > {temp_threshold_hot}°C). "
            f"Bitte passe die Schwellenwerte in der Sidebar an."
        )
    else:
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Kalt (T2M < {temp_threshold_cold}°C) - {len(df_cold)} Punkte",
                f"Heiß (T2M > {temp_threshold_hot}°C) - {len(df_hot)} Punkte",
            ),
            specs=[[{"type": "scattermapbox"}, {"type": "scattermapbox"}]],
            horizontal_spacing=0.05,
        )

        # Calculate center and zoom for map
        all_lats = df_filtered["location_lat"].dropna()
        all_lons = df_filtered["location_long"].dropna()
        center_lat = float(all_lats.mean()) if not all_lats.empty else 0.0
        center_lon = float(all_lons.mean()) if not all_lons.empty else 0.0

        # Cold plot
        if not df_cold.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=df_cold["location_lat"],
                    lon=df_cold["location_long"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        opacity=0.6,
                        color="blue",
                        symbol="circle",
                    ),
                    name="Kalt",
                    hovertemplate="<b>Kalt</b><br>"
                    + "Lat: %{lat:.4f}<br>"
                    + "Lon: %{lon:.4f}<br>"
                    + "T2M: %{customdata:.2f}°C<extra></extra>",
                    customdata=df_cold["T2M"],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Hot plot
        if not df_hot.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=df_hot["location_lat"],
                    lon=df_hot["location_long"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        opacity=0.6,
                        color="red",
                        symbol="circle",
                    ),
                    name="Heiß",
                    hovertemplate="<b>Heiß</b><br>"
                    + "Lat: %{lat:.4f}<br>"
                    + "Lon: %{lon:.4f}<br>"
                    + "T2M: %{customdata:.2f}°C<extra></extra>",
                    customdata=df_hot["T2M"],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=600,
            mapbox1=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=6,
            ),
            mapbox2=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=6,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        # Additional density visualization using Plotly density_mapbox
        if not df_cold.empty or not df_hot.empty:
            st.subheader("Density Heatmap (Alternative Ansicht)")
            density_tab1, density_tab2 = st.tabs(["Kalt", "Heiß"])

            with density_tab1:
                if not df_cold.empty:
                    fig_density_cold = px.density_mapbox(
                        df_cold,
                        lat="location_lat",
                        lon="location_long",
                        z="T2M",
                        radius=10,
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=6,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Blues",
                        title=f"Density Heatmap: Kalt (T2M < {temp_threshold_cold}°C)",
                        height=600,
                    )
                    fig_density_cold.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(
                        fig_density_cold, use_container_width=True, config={"scrollZoom": True}
                    )
                else:
                    st.info("Keine kalten Datenpunkte vorhanden.")

            with density_tab2:
                if not df_hot.empty:
                    fig_density_hot = px.density_mapbox(
                        df_hot,
                        lat="location_lat",
                        lon="location_long",
                        z="T2M",
                        radius=10,
                        center=dict(lat=center_lat, lon=center_lon),
                        zoom=6,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Reds",
                        title=f"Density Heatmap: Heiß (T2M > {temp_threshold_hot}°C)",
                        height=600,
                    )
                    fig_density_hot.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(
                        fig_density_hot, use_container_width=True, config={"scrollZoom": True}
                    )
                else:
                    st.info("Keine heißen Datenpunkte vorhanden.")

# ============================================================================
# TAB 2: Scatter Map with Temperature Coloring
# ============================================================================
with tab2:
    st.subheader("Scatter Map: Temperatur-Färbung")
    st.caption(
        "Die Farbe der Punkte repräsentiert die Temperatur (T2M). "
        "Blau = kalt, Rot = heiß. Zoome hinein, um zu sehen, ob sich bestimmte Farben an bestimmten Orten häufen."
    )

    # Create scatter map with temperature coloring
    fig_scatter = px.scatter_mapbox(
        df_filtered,
        lat="location_lat",
        lon="location_long",
        color="T2M",
        color_continuous_scale="Bluered",  # Blue (kalt) to Red (heiß)
        range_color=[df_filtered["T2M"].min(), df_filtered["T2M"].max()],
        zoom=6,
        height=700,
        hover_data=["timestamp", "individual_local_identifier", "PRECTOTCORR"],
        labels={"T2M": "Temperatur (°C)"},
        title="GPS-Punkte nach Temperatur gefärbt",
    )

    center_lat = float(df_filtered["location_lat"].dropna().mean())
    center_lon = float(df_filtered["location_long"].dropna().mean())

    fig_scatter.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=6),
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title="Temperatur (°C)",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
        ),
    )

    fig_scatter.update_traces(marker=dict(size=5, opacity=0.7))

    st.plotly_chart(fig_scatter, use_container_width=True, config={"scrollZoom": True})

    # Statistics
    st.subheader("Statistiken")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gesamt-Punkte", f"{len(df_filtered):,}")
    with col2:
        st.metric("Temperatur-Bereich", f"{df_filtered['T2M'].min():.1f} - {df_filtered['T2M'].max():.1f}°C")
    with col3:
        st.metric("Standardabweichung", f"{df_filtered['T2M'].std():.2f}°C")

# ============================================================================
# TAB 3: Hexbin Plot
# ============================================================================
with tab3:
    st.subheader("Hexbin Plot: Häufigkeitsverteilung")
    st.caption(
        "Statische Visualisierung der Aufenthaltshäufigkeit in hexagonalen Rastern. "
        "Dies vermeidet Overplotting bei vielen Datenpunkten."
    )

    # Hexbin parameters
    hexbin_col1, hexbin_col2 = st.columns(2)
    with hexbin_col1:
        gridsize = st.slider(
            "Grid-Größe",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Größere Werte = feinere Rasterung",
        )
    with hexbin_col2:
        mincnt = st.slider(
            "Minimale Anzahl Punkte pro Hex",
            min_value=0,
            max_value=50,
            value=1,
            step=1,
            help="Hexagone mit weniger Punkten werden nicht angezeigt",
        )

    # Create hexbin plot
    fig_hex, ax = plt.subplots(figsize=(12, 8))

    # Extract coordinates
    lons = df_filtered["location_long"].values
    lats = df_filtered["location_lat"].values

    # Create hexbin plot
    hb = ax.hexbin(
        lons,
        lats,
        gridsize=gridsize,
        mincnt=mincnt,
        cmap="YlOrRd",  # Yellow-Orange-Red colormap
        linewidths=0.3,
        edgecolors="gray",
    )

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(
        f"Hexbin Plot: Aufenthaltshäufigkeit der Zebras\n"
        f"(Grid-Größe: {gridsize}, Min. Punkte/Hex: {mincnt})",
        fontsize=14,
    )

    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Anzahl GPS-Punkte pro Hexagon", fontsize=11)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig_hex, use_container_width=True)

    # Additional hexbin with temperature coloring
    st.subheader("Hexbin Plot: Nach Temperatur gewichtet")
    st.caption("Die Farbe repräsentiert die durchschnittliche Temperatur in jedem Hexagon.")

    fig_hex_temp, ax_temp = plt.subplots(figsize=(12, 8))

    # Group by hexagon and calculate mean temperature
    # We'll use a simple approach: bin the data
    lon_bins = np.linspace(lons.min(), lons.max(), gridsize + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), gridsize + 1)

    # Create hexbin with temperature as C (values)
    hb_temp = ax_temp.hexbin(
        lons,
        lats,
        C=df_filtered["T2M"].values,
        gridsize=gridsize,
        mincnt=mincnt,
        cmap="coolwarm",  # Blue (kalt) to Red (heiß)
        linewidths=0.3,
        edgecolors="gray",
        reduce_C_function=np.mean,  # Use mean temperature per hexagon
    )

    ax_temp.set_xlabel("Longitude", fontsize=12)
    ax_temp.set_ylabel("Latitude", fontsize=12)
    ax_temp.set_title(
        f"Hexbin Plot: Durchschnittliche Temperatur pro Hexagon\n"
        f"(Grid-Größe: {gridsize}, Min. Punkte/Hex: {mincnt})",
        fontsize=14,
    )

    cb_temp = plt.colorbar(hb_temp, ax=ax_temp)
    cb_temp.set_label("Durchschnittliche Temperatur (°C)", fontsize=11)

    ax_temp.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig_hex_temp, use_container_width=True)

    # Statistics
    st.subheader("Hexbin-Statistiken")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gesamt-Punkte", f"{len(df_filtered):,}")
        st.metric("Grid-Größe", f"{gridsize}x{gridsize}")
    with col2:
        # Count non-empty hexagons
        counts = hb.get_array()
        non_empty = np.sum(counts >= mincnt) if counts is not None else 0
        st.metric("Belegte Hexagone", f"{non_empty:,}")
        if counts is not None and len(counts) > 0:
            st.metric("Max. Punkte pro Hex", f"{int(counts.max()):,}")

st.divider()
st.caption(
    "💡 Tipp: Nutze die Sidebar, um Jahr, Zebra-IDs und Temperatur-Schwellenwerte anzupassen. "
    "Die Visualisierungen werden automatisch aktualisiert."
)
