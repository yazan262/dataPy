from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Pearson-Korrelation (Fallback falls scipy nicht verfügbar)
try:
    from scipy.stats import pearsonr
except ImportError:
    def pearsonr(x, y):
        """Pearson-Korrelation mit numpy"""
        x = np.array(x)
        y = np.array(y)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2:
            return np.nan, np.nan
        x_clean = x[mask]
        y_clean = y[mask]
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        # Einfache p-Wert-Schätzung (nicht exakt, aber ausreichend)
        n = len(x_clean)
        if n < 3:
            p = np.nan
        else:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2)) if abs(corr) < 1 else np.inf
            # Vereinfachte p-Wert-Berechnung
            p = 2 * (1 - abs(t_stat) / (abs(t_stat) + 1)) if np.isfinite(t_stat) else 0.0
        return corr, p

st.set_page_config(page_title="Strecke & Wetter Analyse", layout="wide")
st.title("📈 Strecke & Wetter: Abhängigkeitsanalyse")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "zebra_weather_cleaned.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Berechnet die Distanz zwischen zwei GPS-Punkten in Kilometern"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


@st.cache_data
def calculate_daily_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die tägliche zurückgelegte Strecke pro Zebra"""
    df = df.copy()
    df = df.sort_values(["individual_local_identifier", "timestamp"])
    df = df.dropna(subset=["location_lat", "location_long"])

    # Berechne Distanz zwischen aufeinanderfolgenden Punkten pro Zebra
    df["prev_lat"] = df.groupby("individual_local_identifier")["location_lat"].shift(1)
    df["prev_long"] = df.groupby("individual_local_identifier")["location_long"].shift(1)

    # Berechne Schritt-Distanz
    df["step_km"] = df.apply(
        lambda row: haversine(
            row["location_long"],
            row["location_lat"],
            row["prev_long"],
            row["prev_lat"],
        )
        if pd.notna(row["prev_lat"]) and pd.notna(row["prev_long"])
        else 0.0,
        axis=1,
    )

    # Erstelle Datum-Spalte
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Tägliche Aggregation pro Zebra
    daily_df = (
        df.groupby(["date", "individual_local_identifier"])
        .agg(
            {
                "step_km": "sum",
                "T2M": "mean",
                "PRECTOTCORR": "mean",
                "RH2M": "mean",
                "WS2M": "mean",
                "T2M_MAX": "max",
                "T2M_MIN": "min",
            }
        )
        .reset_index()
    )

    # Umbenennen für Klarheit
    daily_df = daily_df.rename(columns={"step_km": "daily_distance_km"})

    return daily_df


df = load_data(DATA_PATH)

# Sidebar Filter
st.sidebar.header("Filter & Einstellungen")

# Year filter
years = sorted(df["timestamp"].dt.year.dropna().unique())
if "strecke_year" not in st.session_state:
    st.session_state.strecke_year = years[0] if years else None

year = st.sidebar.selectbox(
    "Jahr",
    years,
    index=(years.index(st.session_state.strecke_year) if st.session_state.strecke_year in years else 0),
    key="strecke_year_select",
)
st.session_state.strecke_year = year

# Filter data for selected year
df_year = df[df["timestamp"].dt.year == year].copy()
zebra_ids_year = sorted(df_year["individual_local_identifier"].dropna().unique())

if not zebra_ids_year:
    st.error(f"Keine Daten für das Jahr {year} gefunden.")
    st.stop()

# Metadaten-Sektion: Jahr-Übersicht
st.subheader("📊 Jahr-Übersicht")
col1, col2, col3, col4 = st.columns(4)

# Berechne Jahr-Metadaten
first_date = df_year["timestamp"].min()
last_date = df_year["timestamp"].max()
unique_dates = df_year["timestamp"].dt.date.nunique()
days_in_year = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
coverage_pct = (unique_dates / days_in_year) * 100

with col1:
    st.metric("Erster Datensatz", first_date.strftime("%Y-%m-%d"))
with col2:
    st.metric("Letzter Datensatz", last_date.strftime("%Y-%m-%d"))
with col3:
    st.metric("Erfasste Tage", f"{unique_dates:,}")
with col4:
    st.metric("Jahresabdeckung", f"{coverage_pct:.1f}%")

# Progress bar für Jahresabdeckung
st.progress(coverage_pct / 100, text=f"Datenabdeckung: {unique_dates} von {days_in_year} Tagen ({coverage_pct:.1f}%)")

# Min/Max-Wetterdaten für das Jahr
st.subheader("🌡️ Wetterdaten-Übersicht (Min/Max)")
weather_col1, weather_col2, weather_col3, weather_col4, weather_col5 = st.columns(5)

# Berechne Min/Max für Wetterparameter
with weather_col1:
    if "T2M" in df_year.columns and df_year["T2M"].notna().any():
        t2m_min = df_year["T2M"].min()
        t2m_max = df_year["T2M"].max()
        st.metric("Temperatur Ø (T2M)", f"{t2m_min:.1f} - {t2m_max:.1f} °C")
    else:
        st.metric("Temperatur Ø (T2M)", "N/A")

with weather_col2:
    if "T2M_MIN" in df_year.columns and df_year["T2M_MIN"].notna().any():
        t2m_min_val = df_year["T2M_MIN"].min()
        st.metric("Temperatur Min", f"{t2m_min_val:.1f} °C")
    else:
        st.metric("Temperatur Min", "N/A")

with weather_col3:
    if "T2M_MAX" in df_year.columns and df_year["T2M_MAX"].notna().any():
        t2m_max_val = df_year["T2M_MAX"].max()
        st.metric("Temperatur Max", f"{t2m_max_val:.1f} °C")
    else:
        st.metric("Temperatur Max", "N/A")

with weather_col4:
    if "PRECTOTCORR" in df_year.columns and df_year["PRECTOTCORR"].notna().any():
        precip_min = df_year["PRECTOTCORR"].min()
        precip_max = df_year["PRECTOTCORR"].max()
        st.metric("Niederschlag (PRECTOTCORR)", f"{precip_min:.2f} - {precip_max:.2f} mm")
    else:
        st.metric("Niederschlag (PRECTOTCORR)", "N/A")

with weather_col5:
    if "RH2M" in df_year.columns and df_year["RH2M"].notna().any():
        rh_min = df_year["RH2M"].min()
        rh_max = df_year["RH2M"].max()
        st.metric("Luftfeuchtigkeit (RH2M)", f"{rh_min:.1f} - {rh_max:.1f} %")
    else:
        st.metric("Luftfeuchtigkeit (RH2M)", "N/A")

# Zweite Zeile für weitere Parameter
weather_col6, weather_col7 = st.columns(2)

with weather_col6:
    if "WS2M" in df_year.columns and df_year["WS2M"].notna().any():
        wind_min = df_year["WS2M"].min()
        wind_max = df_year["WS2M"].max()
        st.metric("Windgeschwindigkeit (WS2M)", f"{wind_min:.2f} - {wind_max:.2f} m/s")
    else:
        st.metric("Windgeschwindigkeit (WS2M)", "N/A")

st.divider()

# Metadaten-Sektion: Zebra-Übersicht
st.subheader("🦓 Zebra-Übersicht (alle Zebras im Jahr)")
zebra_metadata = []

for zebra_id in zebra_ids_year:
    df_zebra = df_year[df_year["individual_local_identifier"] == zebra_id].copy()
    df_zebra = df_zebra.dropna(subset=["location_lat", "location_long"])

    if len(df_zebra) > 0:
        zebra_first = df_zebra["timestamp"].min()
        zebra_last = df_zebra["timestamp"].max()
        zebra_days = df_zebra["timestamp"].dt.date.nunique()
        zebra_points = len(df_zebra)

        zebra_metadata.append(
            {
                "Zebra-ID": zebra_id,
                "Erster Tag": zebra_first.strftime("%Y-%m-%d"),
                "Letzter Tag": zebra_last.strftime("%Y-%m-%d"),
                "Tracking-Tage": zebra_days,
                "GPS-Punkte": zebra_points,
            }
        )

if zebra_metadata:
    df_zebra_meta = pd.DataFrame(zebra_metadata)
    st.dataframe(df_zebra_meta, use_container_width=True, hide_index=True)
else:
    st.warning("Keine Zebra-Daten gefunden.")

st.divider()

# Zebra-Auswahl
if "strecke_zebra_ids" not in st.session_state:
    st.session_state.strecke_zebra_ids = []

# Select All / Deselect All Buttons
col_select1, col_select2 = st.sidebar.columns(2)
with col_select1:
    if st.button("✓ Alle auswählen", use_container_width=True):
        st.session_state.strecke_zebra_ids = zebra_ids_year.copy()
        st.session_state.strecke_zebra_select = zebra_ids_year.copy()
        st.rerun()

with col_select2:
    if st.button("✗ Alle abwählen", use_container_width=True):
        st.session_state.strecke_zebra_ids = []
        st.session_state.strecke_zebra_select = []
        st.rerun()

zebra_ids_selected = st.sidebar.multiselect(
    "Zebra-IDs (aus gewähltem Jahr)",
    zebra_ids_year,
    default=st.session_state.strecke_zebra_ids,
    key="strecke_zebra_select",
)
st.session_state.strecke_zebra_ids = zebra_ids_selected

# Wetterparameter-Auswahl
st.sidebar.divider()
st.sidebar.subheader("Wetterparameter")
show_t2m = st.sidebar.checkbox("Temperatur Ø (T2M)", value=True)
show_t2m_max = st.sidebar.checkbox("Temperatur Max (T2M_MAX)", value=False)
show_precip = st.sidebar.checkbox("Niederschlag (PRECTOTCORR)", value=True)
show_humidity = st.sidebar.checkbox("Luftfeuchtigkeit (RH2M)", value=False)
show_wind = st.sidebar.checkbox("Windgeschwindigkeit (WS2M)", value=False)

show_moving_avg = st.sidebar.checkbox("7-Tage Moving Average anzeigen", value=False)

if not zebra_ids_selected:
    st.warning("Bitte wähle mindestens eine Zebra-ID aus.")
    st.stop()

# Filter data for selected zebras
df_filtered = df_year[df_year["individual_local_identifier"].isin(zebra_ids_selected)].copy()
df_filtered = df_filtered.dropna(subset=["location_lat", "location_long"])

if df_filtered.empty:
    st.error("Keine Daten für die ausgewählten Zebras vorhanden.")
    st.stop()

# Berechne tägliche Strecken
df_daily = calculate_daily_distances(df_filtered)

# Jahreszeiten-Auswahl (auf der Hauptseite)
st.divider()
st.subheader("📅 Jahreszeiten-Filter")
season_option = st.radio(
    "Wähle eine Jahreszeit für die Analyse:",
    ["Ganzes Jahr", "Trockenzeit (Mai-Oktober)", "Regenzeit (November-April)"],
    horizontal=True,
    key="season_filter",
    help="Filtert die Daten nach Jahreszeit, um ähnliche Perioden miteinander zu vergleichen (z.B. nur Sommer-Daten)"
)

# Berechne Metadaten für beide Jahreszeiten (basierend auf df_year, nicht df_daily)
df_year_dry = df_year[df_year["timestamp"].dt.month.isin([5, 6, 7, 8, 9, 10])].copy()
df_year_wet = df_year[df_year["timestamp"].dt.month.isin([11, 12, 1, 2, 3, 4])].copy()

# Metadaten: Temperatur nach Jahreszeiten
st.subheader("🌡️ Temperatur-Metadaten nach Jahreszeiten")

meta_col1, meta_col2 = st.columns(2)

with meta_col1:
    # Erwarteter Datumsbereich für Trockenzeit
    expected_dry_start = pd.Timestamp(year, 5, 1)  # Mai des Jahres
    expected_dry_end = pd.Timestamp(year, 10, 31)  # Oktober des Jahres
    expected_dry_range = f"{expected_dry_start.strftime('%b %Y')} - {expected_dry_end.strftime('%b %Y')}"
    
    st.markdown(f"**Trockenzeit (Mai-Oktober)**")
    st.caption(f"📅 Erwartet: {expected_dry_range}")
    
    if not df_year_dry.empty:
        # Tatsächlicher Datumsbereich der vorhandenen Daten
        actual_dry_start = df_year_dry["timestamp"].min()
        actual_dry_end = df_year_dry["timestamp"].max()
        actual_dry_range = f"{actual_dry_start.strftime('%b %Y')} - {actual_dry_end.strftime('%b %Y')}"
        if actual_dry_range != expected_dry_range:
            st.caption(f"⚠️ Vorhanden: {actual_dry_range}")
    
    if not df_year_dry.empty:
        # Min: Verwende T2M_MIN (Minimum-Temperatur)
        if "T2M_MIN" in df_year_dry.columns and df_year_dry["T2M_MIN"].notna().any():
            t2m_dry_min = df_year_dry["T2M_MIN"].min()
        elif "T2M" in df_year_dry.columns and df_year_dry["T2M"].notna().any():
            t2m_dry_min = df_year_dry["T2M"].min()
        else:
            t2m_dry_min = None
        
        # Max: Verwende T2M_MAX (Maximum-Temperatur) - das ist wichtig!
        if "T2M_MAX" in df_year_dry.columns and df_year_dry["T2M_MAX"].notna().any():
            t2m_dry_max = df_year_dry["T2M_MAX"].max()
        elif "T2M" in df_year_dry.columns and df_year_dry["T2M"].notna().any():
            t2m_dry_max = df_year_dry["T2M"].max()
        else:
            t2m_dry_max = None
        
        # Durchschnitt: Verwende T2M (Durchschnittstemperatur)
        if "T2M" in df_year_dry.columns and df_year_dry["T2M"].notna().any():
            t2m_dry_avg = df_year_dry["T2M"].mean()
        else:
            t2m_dry_avg = None
        
        if t2m_dry_min is not None:
            st.metric("Min Temperatur", f"{t2m_dry_min:.1f} °C")
        if t2m_dry_max is not None:
            st.metric("Max Temperatur", f"{t2m_dry_max:.1f} °C")
        if t2m_dry_avg is not None:
            st.metric("Durchschnitt", f"{t2m_dry_avg:.1f} °C")
        st.caption(f"📊 {len(df_year_dry)} Datenpunkte")
    else:
        st.info("Keine Daten verfügbar")

with meta_col2:
    # Erwarteter Datumsbereich für Regenzeit (Nov/Dez des Vorjahres + Jan-Apr des aktuellen Jahres)
    expected_wet_start = pd.Timestamp(year - 1, 11, 1)  # November des Vorjahres
    expected_wet_end = pd.Timestamp(year, 4, 30)  # April des aktuellen Jahres
    expected_wet_range = f"{expected_wet_start.strftime('%b %Y')} - {expected_wet_end.strftime('%b %Y')}"
    
    st.markdown(f"**Regenzeit (November-April)**")
    st.caption(f"📅 Erwartet: {expected_wet_range}")
    
    if not df_year_wet.empty:
        # Tatsächlicher Datumsbereich der vorhandenen Daten
        actual_wet_start = df_year_wet["timestamp"].min()
        actual_wet_end = df_year_wet["timestamp"].max()
        actual_wet_range = f"{actual_wet_start.strftime('%b %Y')} - {actual_wet_end.strftime('%b %Y')}"
        if actual_wet_range != expected_wet_range:
            st.caption(f"⚠️ Vorhanden: {actual_wet_range}")
    
    if not df_year_wet.empty:
        # Min: Verwende T2M_MIN (Minimum-Temperatur)
        if "T2M_MIN" in df_year_wet.columns and df_year_wet["T2M_MIN"].notna().any():
            t2m_wet_min = df_year_wet["T2M_MIN"].min()
        elif "T2M" in df_year_wet.columns and df_year_wet["T2M"].notna().any():
            t2m_wet_min = df_year_wet["T2M"].min()
        else:
            t2m_wet_min = None
        
        # Max: Verwende T2M_MAX (Maximum-Temperatur) - das ist wichtig!
        if "T2M_MAX" in df_year_wet.columns and df_year_wet["T2M_MAX"].notna().any():
            t2m_wet_max = df_year_wet["T2M_MAX"].max()
        elif "T2M" in df_year_wet.columns and df_year_wet["T2M"].notna().any():
            t2m_wet_max = df_year_wet["T2M"].max()
        else:
            t2m_wet_max = None
        
        # Durchschnitt: Verwende T2M (Durchschnittstemperatur)
        if "T2M" in df_year_wet.columns and df_year_wet["T2M"].notna().any():
            t2m_wet_avg = df_year_wet["T2M"].mean()
        else:
            t2m_wet_avg = None
        
        if t2m_wet_min is not None:
            st.metric("Min Temperatur", f"{t2m_wet_min:.1f} °C")
        if t2m_wet_max is not None:
            st.metric("Max Temperatur", f"{t2m_wet_max:.1f} °C")
        if t2m_wet_avg is not None:
            st.metric("Durchschnitt", f"{t2m_wet_avg:.1f} °C")
        st.caption(f"📊 {len(df_year_wet)} Datenpunkte")
    else:
        st.info("Keine Daten verfügbar")

st.divider()

# Filtere Daten nach Jahreszeit für die Visualisierung
if season_option == "Trockenzeit (Mai-Oktober)":
    df_daily = df_daily[df_daily["date"].dt.month.isin([5, 6, 7, 8, 9, 10])].copy()
    season_label = "Trockenzeit"
elif season_option == "Regenzeit (November-April)":
    df_daily = df_daily[df_daily["date"].dt.month.isin([11, 12, 1, 2, 3, 4])].copy()
    season_label = "Regenzeit"
else:
    season_label = "Ganzes Jahr"

if df_daily.empty:
    st.warning(f"Keine Daten für die ausgewählte Jahreszeit ({season_label}) vorhanden.")
    st.stop()

st.caption(f"📊 Aktuell angezeigt: {season_label} - {len(df_daily)} Tage")

st.divider()

# Zebra-Selektor für einzelne Ansicht
st.subheader("Zebra-Auswahl für detaillierte Analyse")
if len(zebra_ids_selected) == 1:
    selected_zebra = zebra_ids_selected[0]
else:
    selected_zebra = st.selectbox(
        "Wähle ein Zebra für detaillierte Analyse:",
        zebra_ids_selected,
        key="zebra_detail_select",
    )

# Filter für ausgewähltes Zebra
df_zebra_daily = df_daily[df_daily["individual_local_identifier"] == selected_zebra].copy()
df_zebra_daily = df_zebra_daily.sort_values("date")

if df_zebra_daily.empty:
    st.error(f"Keine täglichen Daten für Zebra {selected_zebra} gefunden.")
    st.stop()

# Metriken-Karten für ausgewähltes Zebra
st.divider()
st.subheader(f"📊 Metriken: {selected_zebra}")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    avg_distance = df_zebra_daily["daily_distance_km"].mean()
    st.metric("Ø Tägliche Strecke", f"{avg_distance:.2f} km")
with metric_col2:
    max_distance = df_zebra_daily["daily_distance_km"].max()
    st.metric("Max. Tägliche Strecke", f"{max_distance:.2f} km")
with metric_col3:
    total_distance = df_zebra_daily["daily_distance_km"].sum()
    st.metric("Gesamtstrecke", f"{total_distance:.1f} km")
with metric_col4:
    tracking_days = len(df_zebra_daily)
    st.metric("Tracking-Tage", f"{tracking_days:,}")

# Korrelationskoeffizienten
st.subheader("🔗 Korrelationen mit Wetterparametern")
corr_col1, corr_col2, corr_col3, corr_col4, corr_col5 = st.columns(5)

correlations = {}
if show_t2m:
    try:
        r, p = pearsonr(
            df_zebra_daily["daily_distance_km"].dropna(),
            df_zebra_daily["T2M"].dropna(),
        )
        correlations["T2M"] = (r, p)
    except:
        correlations["T2M"] = (np.nan, np.nan)

if show_t2m_max:
    try:
        r, p = pearsonr(
            df_zebra_daily["daily_distance_km"].dropna(),
            df_zebra_daily["T2M_MAX"].dropna(),
        )
        correlations["T2M_MAX"] = (r, p)
    except:
        correlations["T2M_MAX"] = (np.nan, np.nan)

if show_precip:
    try:
        r, p = pearsonr(
            df_zebra_daily["daily_distance_km"].dropna(),
            df_zebra_daily["PRECTOTCORR"].dropna(),
        )
        correlations["PRECTOTCORR"] = (r, p)
    except:
        correlations["PRECTOTCORR"] = (np.nan, np.nan)

if show_humidity:
    try:
        r, p = pearsonr(
            df_zebra_daily["daily_distance_km"].dropna(),
            df_zebra_daily["RH2M"].dropna(),
        )
        correlations["RH2M"] = (r, p)
    except:
        correlations["RH2M"] = (np.nan, np.nan)

if show_wind:
    try:
        r, p = pearsonr(
            df_zebra_daily["daily_distance_km"].dropna(),
            df_zebra_daily["WS2M"].dropna(),
        )
        correlations["WS2M"] = (r, p)
    except:
        correlations["WS2M"] = (np.nan, np.nan)

corr_labels = {
    "T2M": "Temperatur Ø",
    "T2M_MAX": "Temperatur Max",
    "PRECTOTCORR": "Niederschlag",
    "RH2M": "Luftfeuchtigkeit",
    "WS2M": "Wind",
}

corr_items = list(correlations.items())
for idx, (param, (r, p)) in enumerate(correlations.items()):
    with [corr_col1, corr_col2, corr_col3, corr_col4, corr_col5][idx % 5]:
        if not np.isnan(r):
            st.metric(
                corr_labels[param],
                f"r = {r:.3f}",
                help=f"p-Wert: {p:.4f}" if not np.isnan(p) else "N/A",
            )
        else:
            st.metric(corr_labels[param], "N/A")

st.divider()

# Tabs für verschiedene Visualisierungen
tab1, tab2, tab3 = st.tabs(["📈 Zeitreihen", "🔍 Scatter-Plots", "📊 Vergleich"])

# ============================================================================
# TAB 1: Zeitreihen-Charts
# ============================================================================
with tab1:
    st.subheader(f"Zeitreihen: {selected_zebra} ({season_label})")

    # Sammle Wetterparameter
    weather_params = []
    if show_t2m:
        weather_params.append(("T2M", "Temperatur Ø (°C)", "red"))
    if show_t2m_max:
        weather_params.append(("T2M_MAX", "Temperatur Max (°C)", "darkred"))
    if show_precip:
        weather_params.append(("PRECTOTCORR", "Niederschlag (mm)", "blue"))
    if show_humidity:
        weather_params.append(("RH2M", "Luftfeuchtigkeit (%)", "green"))
    if show_wind:
        weather_params.append(("WS2M", "Windgeschwindigkeit (m/s)", "orange"))

    # Anzahl der Subplots: 1 für Strecke + Anzahl Wetterparameter
    n_plots = 1 + len(weather_params)
    
    # Erstelle Subplots - jeder Parameter bekommt einen eigenen
    subplot_titles = ["Tägliche zurückgelegte Strecke"]
    subplot_titles.extend([f"{label}" for _, label, _ in weather_params])
    
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    # Plot 1: Tägliche Strecke
    fig.add_trace(
        go.Scatter(
            x=df_zebra_daily["date"],
            y=df_zebra_daily["daily_distance_km"],
            mode="lines+markers",
            name="Tägliche Strecke",
            line=dict(color="blue", width=2),
            marker=dict(size=4, opacity=0.7),
            hovertemplate="<b>%{fullData.name}</b><br>"
            + "Datum: %{x}<br>"
            + "Strecke: %{y:.2f} km<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Moving Average für Strecke
    if show_moving_avg:
        df_zebra_daily_sorted = df_zebra_daily.sort_values("date")
        moving_avg = (
            df_zebra_daily_sorted["daily_distance_km"]
            .rolling(window=7, center=True)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=df_zebra_daily_sorted["date"],
                y=moving_avg,
                mode="lines",
                name="7-Tage Durchschnitt",
                line=dict(color="darkblue", width=2, dash="dash"),
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Datum: %{x}<br>"
                + "Strecke: %{y:.2f} km<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Jeder Wetterparameter in einem eigenen Subplot
    for idx, (param, label, color) in enumerate(weather_params, start=2):
        fig.add_trace(
            go.Scatter(
                x=df_zebra_daily["date"],
                y=df_zebra_daily[param],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=4, opacity=0.7),
                hovertemplate=f"<b>{label}</b><br>"
                + "Datum: %{x}<br>"
                + f"{label}: %{{y:.2f}}<extra></extra>",
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

    # Update Layout
    fig.update_layout(
        height=300 * n_plots,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Y-Achsen-Beschriftungen
    fig.update_yaxes(title_text="Strecke (km)", row=1, col=1)
    for idx, (_, label, _) in enumerate(weather_params, start=2):
        fig.update_yaxes(title_text=label, row=idx, col=1)
    
    # X-Achsen-Beschriftung nur im letzten Subplot
    fig.update_xaxes(title_text="Datum", row=n_plots, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Dual-Axis Option (Strecke + Temperatur zusammen)
    if show_t2m:
        st.subheader("Dual-Axis: Strecke & Temperatur")
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

        fig_dual.add_trace(
            go.Scatter(
                x=df_zebra_daily["date"],
                y=df_zebra_daily["daily_distance_km"],
                mode="lines+markers",
                name="Tägliche Strecke",
                line=dict(color="blue", width=2),
            ),
            secondary_y=False,
        )

        fig_dual.add_trace(
            go.Scatter(
                x=df_zebra_daily["date"],
                y=df_zebra_daily["T2M"],
                mode="lines+markers",
                name="Temperatur",
                line=dict(color="red", width=2),
            ),
            secondary_y=True,
        )

        fig_dual.update_xaxes(title_text="Datum")
        fig_dual.update_yaxes(title_text="Strecke (km)", secondary_y=False)
        fig_dual.update_yaxes(title_text="Temperatur (°C)", secondary_y=True)

        fig_dual.update_layout(
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_dual, use_container_width=True)

# ============================================================================
# TAB 2: Scatter-Plots
# ============================================================================
with tab2:
    st.subheader(f"Scatter-Plots: {selected_zebra} ({season_label})")

    # Erklärungssektion
    with st.expander("📖 Wie liest man diese Scatter-Plots?", expanded=False):
        st.markdown("""
        **Was zeigt dieser Plot?**
        
        Die Scatter-Plots zeigen die Beziehung zwischen einem Wetterparameter (X-Achse) und der täglich zurückgelegten Strecke (Y-Achse) für das ausgewählte Zebra.
        Jeder Punkt im Plot repräsentiert einen einzelnen Tag mit:
        - **X-Achse:** Wert des Wetterparameters an diesem Tag
        - **Y-Achse:** Zurückgelegte Strecke an diesem Tag (in Kilometern)
        
        **Wie versteht man die Visualisierung?**
        
        - **Punktewolke:** Die Verteilung der Punkte zeigt, ob es einen Zusammenhang gibt
          - Punkte steigen nach rechts → positive Korrelation (mehr Wetter = mehr Strecke)
          - Punkte fallen nach rechts → negative Korrelation (mehr Wetter = weniger Strecke)
          - Punkte zufällig verteilt → keine klare Beziehung
        
        - **Korrelationskoeffizient (r):** 
          - **r nahe +1:** Starke positive Korrelation (z.B. höhere Temperatur → mehr Bewegung)
          - **r nahe -1:** Starke negative Korrelation (z.B. mehr Regen → weniger Bewegung)
          - **r nahe 0:** Keine oder sehr schwache Korrelation
          - **p-Wert:** Statistische Signifikanz (p < 0.05 = signifikant)
        
        **Einheiten:**
        - **Tägliche Strecke:** Kilometer (km) - Summe aller GPS-Schritte pro Tag
        - **Temperatur Ø (T2M):** Grad Celsius (°C) - Durchschnittstemperatur am Tag
        - **Temperatur Max (T2M_MAX):** Grad Celsius (°C) - Maximale Temperatur am Tag
        - **Niederschlag (PRECTOTCORR):** Millimeter (mm) - Gesamtniederschlag pro Tag
        - **Luftfeuchtigkeit (RH2M):** Prozent (%) - Relative Luftfeuchtigkeit
        - **Windgeschwindigkeit (WS2M):** Meter pro Sekunde (m/s) - Windgeschwindigkeit in 2m Höhe
        """)

    scatter_params = []
    if show_t2m:
        scatter_params.append(("T2M", "Temperatur Ø (°C)", "red"))
    if show_t2m_max:
        scatter_params.append(("T2M_MAX", "Temperatur Max (°C)", "darkred"))
    if show_precip:
        scatter_params.append(("PRECTOTCORR", "Niederschlag (mm)", "blue"))
    if show_humidity:
        scatter_params.append(("RH2M", "Luftfeuchtigkeit (%)", "green"))
    if show_wind:
        scatter_params.append(("WS2M", "Windgeschwindigkeit (m/s)", "orange"))

    if not scatter_params:
        st.info("Bitte wähle mindestens einen Wetterparameter in der Sidebar aus.")
    else:
        st.caption(
            "💡 **Hinweis:** Jeder Punkt repräsentiert einen Tag. Die X-Achse zeigt den Wetterwert, "
            "die Y-Achse die zurückgelegte Strecke in km. Der Korrelationskoeffizient (r) zeigt die Stärke "
            "und Richtung der Beziehung. Klicke auf '📖 Wie liest man diese Scatter-Plots?' oben für Details."
        )
        
        # Erstelle Subplots für Scatter-Plots
        n_params = len(scatter_params)
        n_cols = 2
        n_rows = (n_params + 1) // 2

        fig_scatter = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Strecke vs. {label}" for _, label, _ in scatter_params],
            vertical_spacing=0.15,
        )

        for idx, (param, label, color) in enumerate(scatter_params):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            # Berechne Korrelation
            try:
                r, p = pearsonr(
                    df_zebra_daily["daily_distance_km"].dropna(),
                    df_zebra_daily[param].dropna(),
                )
                corr_text = f"r = {r:.3f}"
                if not np.isnan(p):
                    corr_text += f" (p = {p:.4f})"
            except:
                r, p = np.nan, np.nan
                corr_text = "r = N/A"

            fig_scatter.add_trace(
                go.Scatter(
                    x=df_zebra_daily[param],
                    y=df_zebra_daily["daily_distance_km"],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=6, opacity=0.6),
                    hovertemplate=f"<b>{label}</b>: %{{x:.2f}}<br>"
                    + "Strecke: %{y:.2f} km<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Füge Korrelations-Text hinzu
            fig_scatter.add_annotation(
                text=corr_text,
                xref=f"x{idx+1}",
                yref=f"y{idx+1}",
                x=0.05,
                y=0.95,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                row=row,
                col=col,
            )

            fig_scatter.update_xaxes(title_text=label, row=row, col=col)
            fig_scatter.update_yaxes(title_text="Tägliche Strecke (km)", row=row, col=col)

        fig_scatter.update_layout(
            height=300 * n_rows,
            showlegend=False,
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================================
# TAB 3: Vergleichsansicht
# ============================================================================
with tab3:
    st.subheader(f"Vergleich: Alle ausgewählten Zebras ({season_label})")

    if len(zebra_ids_selected) == 1:
        st.info("Wähle mehrere Zebras aus, um sie zu vergleichen.")
    else:
        # Farbpalette für Zebras
        color_palette = px.colors.qualitative.Plotly
        zebra_colors = {
            zebra_id: color_palette[i % len(color_palette)]
            for i, zebra_id in enumerate(zebra_ids_selected)
        }

        # Zeitreihen-Vergleich
        fig_compare = go.Figure()

        for zebra_id in zebra_ids_selected:
            df_zebra_comp = df_daily[df_daily["individual_local_identifier"] == zebra_id].copy()
            df_zebra_comp = df_zebra_comp.sort_values("date")

            fig_compare.add_trace(
                go.Scatter(
                    x=df_zebra_comp["date"],
                    y=df_zebra_comp["daily_distance_km"],
                    mode="lines+markers",
                    name=zebra_id,
                    line=dict(color=zebra_colors[zebra_id], width=2),
                    marker=dict(size=4, opacity=0.7),
                    hovertemplate=f"<b>{zebra_id}</b><br>"
                    + "Datum: %{x}<br>"
                    + "Strecke: %{y:.2f} km<extra></extra>",
                )
            )

        fig_compare.update_layout(
            title="Tägliche Strecke: Vergleich aller Zebras",
            xaxis_title="Datum",
            yaxis_title="Tägliche Strecke (km)",
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # Korrelations-Vergleich
        st.subheader("Korrelations-Vergleich")
        comparison_corrs = []

        for zebra_id in zebra_ids_selected:
            df_zebra_comp = df_daily[df_daily["individual_local_identifier"] == zebra_id].copy()
            zebra_corrs = {"Zebra-ID": zebra_id}

            for param, label in corr_labels.items():
                if param in df_zebra_comp.columns:
                    try:
                        r, _ = pearsonr(
                            df_zebra_comp["daily_distance_km"].dropna(),
                            df_zebra_comp[param].dropna(),
                        )
                        zebra_corrs[label] = f"{r:.3f}" if not np.isnan(r) else "N/A"
                    except:
                        zebra_corrs[label] = "N/A"

            comparison_corrs.append(zebra_corrs)

        if comparison_corrs:
            df_corr_compare = pd.DataFrame(comparison_corrs)
            st.dataframe(df_corr_compare, use_container_width=True, hide_index=True)

            # Visualisiere Korrelationen als Heatmap
            st.subheader("Korrelations-Heatmap")
            df_corr_plot = df_corr_compare.set_index("Zebra-ID")
            # Konvertiere zu numerischen Werten
            for col in df_corr_plot.columns:
                df_corr_plot[col] = pd.to_numeric(df_corr_plot[col], errors="coerce")

            fig_heatmap = px.imshow(
                df_corr_plot.T,
                labels=dict(x="Zebra-ID", y="Wetterparameter", color="Korrelation"),
                color_continuous_scale="RdBu",
                aspect="auto",
                title="Korrelationskoeffizienten: Strecke vs. Wetter",
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()
st.caption(
    "💡 Tipp: Nutze die Sidebar, um Jahr, Zebras und Wetterparameter anzupassen. "
    "Die Visualisierungen werden automatisch aktualisiert."
)
