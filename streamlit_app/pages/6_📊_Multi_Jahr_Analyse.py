from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

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

st.set_page_config(page_title="Multi-Jahr Strecke-Wetter Analyse", layout="wide")
st.title("📊 Multi-Jahr Strecke & Wetter: Abhängigkeitsanalyse")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "zebra_weather_cleaned.csv"

# Vordefinierte Zeiträume
PREDEFINED_PERIODS = {
    "Mai 2008 - Apr 2009 (empfohlen)": {
        "start": pd.Timestamp(2008, 5, 1),
        "end": pd.Timestamp(2009, 4, 30),
        "description": "Vollständige Trockenzeit (Mai-Okt 2008) + Regenzeit (Nov-Dez 2008 + Jan-Apr 2009)"
    },
    "Nov 2007 - Okt 2008": {
        "start": pd.Timestamp(2007, 11, 1),
        "end": pd.Timestamp(2008, 10, 31),
        "description": "Regenzeit (Nov-Dez 2007 + Jan-Apr 2008) + Trockenzeit (Mai-Okt 2008)"
    },
    "Nov 2008 - Jun 2009": {
        "start": pd.Timestamp(2008, 11, 1),
        "end": pd.Timestamp(2009, 6, 1),
        "description": "Regenzeit (Nov-Dez 2008 + Jan-Apr 2009) + Trockenzeit teilweise (Mai-Jun 2009)"
    }
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Lädt die Zebra-Wetter-Daten"""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
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


def find_common_zebras(df: pd.DataFrame, years: List[int], season: Optional[str] = None) -> pd.DataFrame:
    """
    Findet Zebras, die in mehreren Jahren getrackt wurden.
    
    Args:
        df: DataFrame mit Zebra-Daten
        years: Liste der Jahre
        season: Optional "Trockenzeit" oder "Regenzeit" für Filterung
    
    Returns:
        DataFrame mit Zebra-IDs und Jahren, in denen sie getrackt wurden
    """
    df_filtered = df[df["year"].isin(years)].copy()
    
    # Filter nach Jahreszeit falls angegeben
    if season == "Trockenzeit":
        df_filtered = df_filtered[df_filtered["month"].isin([5, 6, 7, 8, 9, 10])]
    elif season == "Regenzeit":
        df_filtered = df_filtered[df_filtered["month"].isin([11, 12, 1, 2, 3, 4])]
    
    # Finde Zebras pro Jahr
    zebra_year_matrix = defaultdict(set)
    for year in years:
        df_year = df_filtered[df_filtered["year"] == year]
        zebras_year = set(df_year["individual_local_identifier"].dropna().unique())
        for zebra_id in zebras_year:
            zebra_year_matrix[zebra_id].add(year)
    
    # Erstelle DataFrame mit gemeinsamen Zebras
    common_zebras = []
    for zebra_id, years_set in zebra_year_matrix.items():
        if len(years_set) > 1:  # Nur Zebras in mehreren Jahren
            common_zebras.append({
                "Zebra-ID": zebra_id,
                "Jahre": sorted(years_set),
                "Anzahl Jahre": len(years_set)
            })
    
    if common_zebras:
        return pd.DataFrame(common_zebras).sort_values("Zebra-ID")
    else:
        return pd.DataFrame(columns=["Zebra-ID", "Jahre", "Anzahl Jahre"])


def robust_mean(values: pd.Series, method: str = "mean") -> float:
    """
    Berechnet robusten Mittelwert mit verschiedenen Methoden.
    
    Args:
        values: Series mit Werten
        method: "mean" (normal), "median" (Median), "trimmed" (getrimmt), "iqr_filtered" (IQR-gefiltert)
    
    Returns:
        Robustes Mittel
    """
    values_clean = values.dropna()
    
    if len(values_clean) == 0:
        return np.nan
    
    if method == "median":
        return values_clean.median()
    
    elif method == "trimmed":
        # Entferne oberste und unterste 10%
        q10 = values_clean.quantile(0.10)
        q90 = values_clean.quantile(0.90)
        trimmed = values_clean[(values_clean >= q10) & (values_clean <= q90)]
        return trimmed.mean() if len(trimmed) > 0 else values_clean.mean()
    
    elif method == "iqr_filtered":
        # IQR-basierte Filterung: entferne Werte außerhalb von Q1-1.5*IQR bis Q3+1.5*IQR
        Q1 = values_clean.quantile(0.25)
        Q3 = values_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered = values_clean[(values_clean >= lower_bound) & (values_clean <= upper_bound)]
        return filtered.mean() if len(filtered) > 0 else values_clean.mean()
    
    else:  # "mean" - normaler Durchschnitt
        return values_clean.mean()


def aggregate_daily_data(
    df_daily: pd.DataFrame,
    mode: str = "average",
    zebras: Optional[List[str]] = None,
    common_days_only: bool = False,
    robust_method: str = "mean"
) -> pd.DataFrame:
    """
    Aggregiert tägliche Daten nach gewähltem Modus (Hybrid-Ansatz).
    
    Args:
        df_daily: DataFrame mit täglichen Daten pro Zebra
        mode: "average" (Durchschnitt), "individual" (Einzelne), "combined" (Kombiniert)
        zebras: Liste der Zebra-IDs (None = alle)
        common_days_only: Nur Tage mit allen Zebras
        robust_method: "mean", "median", "trimmed", "iqr_filtered"
    
    Returns:
        Aggregierter DataFrame
    """
    if zebras:
        df_daily = df_daily[df_daily["individual_local_identifier"].isin(zebras)].copy()
    
    if df_daily.empty:
        return pd.DataFrame()
    
    if mode == "average":
        # Durchschnittswerte pro Tag
        if common_days_only and zebras:
            # Nur Tage mit allen Zebras
            dates_with_all = df_daily.groupby("date")["individual_local_identifier"].nunique()
            dates_with_all = dates_with_all[dates_with_all == len(zebras)].index
            df_daily = df_daily[df_daily["date"].isin(dates_with_all)]
        
        # Aggregation mit robustem Mittelwert für Strecke
        aggregated_list = []
        for date in df_daily["date"].unique():
            df_date = df_daily[df_daily["date"] == date]
            
            aggregated_list.append({
                "date": date,
                "daily_distance_km": robust_mean(df_date["daily_distance_km"], robust_method),
                "T2M": df_date["T2M"].mean(),  # Wetterparameter bleiben normal
                "PRECTOTCORR": df_date["PRECTOTCORR"].mean(),
                "RH2M": df_date["RH2M"].mean(),
                "WS2M": df_date["WS2M"].mean(),
                "T2M_MAX": df_date["T2M_MAX"].mean(),
                "T2M_MIN": df_date["T2M_MIN"].mean(),
            })
        
        aggregated = pd.DataFrame(aggregated_list)
        aggregated["zebra_id"] = "Durchschnitt"
        return aggregated
    
    elif mode == "individual":
        # Einzelne Zebras separat
        if common_days_only and zebras:
            dates_with_all = df_daily.groupby("date")["individual_local_identifier"].nunique()
            dates_with_all = dates_with_all[dates_with_all == len(zebras)].index
            df_daily = df_daily[df_daily["date"].isin(dates_with_all)]
        
        return df_daily.copy()
    
    elif mode == "combined":
        # Kombiniert: Durchschnitt + Einzelne
        avg_df = aggregate_daily_data(df_daily, "average", zebras, common_days_only)
        ind_df = aggregate_daily_data(df_daily, "individual", zebras, common_days_only)
        
        if avg_df.empty:
            return ind_df
        if ind_df.empty:
            return avg_df
        
        return pd.concat([avg_df, ind_df], ignore_index=True)
    
    else:
        return df_daily.copy()


def get_season_ranges(years: List[int]) -> Dict[str, Dict]:
    """
    Berechnet korrekte Datumsbereiche für Jahreszeiten über Jahre hinweg.
    
    Returns:
        Dict mit "Trockenzeit" und "Regenzeit", jeweils mit "start" und "end"
    """
    ranges = {
        "Trockenzeit": {
            "start": pd.Timestamp(min(years), 5, 1),
            "end": pd.Timestamp(max(years), 10, 31),
            "months": [5, 6, 7, 8, 9, 10]
        },
        "Regenzeit": {
            "start": pd.Timestamp(min(years) - 1, 11, 1),
            "end": pd.Timestamp(max(years), 4, 30),
            "months": [11, 12, 1, 2, 3, 4]
        }
    }
    return ranges


def filter_by_season(df: pd.DataFrame, season: str, years: List[int]) -> pd.DataFrame:
    """
    Filtert Daten nach Jahreszeit, berücksichtigt Jahresgrenzen.
    
    Args:
        df: DataFrame mit Daten
        season: "Trockenzeit" oder "Regenzeit"
        years: Liste der Jahre
    
    Returns:
        Gefilterter DataFrame
    """
    if season == "Trockenzeit":
        # Trockenzeit: Mai-Okt innerhalb eines Jahres
        return df[(df["month"].isin([5, 6, 7, 8, 9, 10])) & (df["year"].isin(years))].copy()
    elif season == "Regenzeit":
        # Regenzeit: Nov-Dez Jahr N-1 + Jan-Apr Jahr N
        df_regen = pd.DataFrame()
        for year in years:
            # Nov-Dez des Vorjahres
            df_nov_dec = df[
                (df["year"] == year - 1) & (df["month"].isin([11, 12]))
            ]
            # Jan-Apr des aktuellen Jahres
            df_jan_apr = df[
                (df["year"] == year) & (df["month"].isin([1, 2, 3, 4]))
            ]
            df_regen = pd.concat([df_regen, df_nov_dec, df_jan_apr], ignore_index=True)
        return df_regen
    else:
        return df.copy()


def calculate_regression_line(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Regressions-Linie für Scatter-Plots.
    
    Args:
        x: X-Werte (Wetterparameter)
        y: Y-Werte (Strecke)
    
    Returns:
        Tuple von (x_values, y_predicted) für die Regressions-Linie
    """
    # Entferne NaN-Werte
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x[mask])
    y_clean = np.array(y[mask])
    
    if len(x_clean) < 2:
        return np.array([]), np.array([])
    
    # Lineare Regression: y = a*x + b
    coeffs = np.polyfit(x_clean, y_clean, 1)
    a, b = coeffs
    
    # Erstelle x-Werte für die Linie (min bis max)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = a * x_line + b
    
    return x_line, y_line


def analyze_trend_by_bins(
    df: pd.DataFrame,
    param: str,
    bins: int = 10,
    bin_width: Optional[float] = None
) -> pd.DataFrame:
    """
    Binning-Analyse: Gruppiert Daten nach Wetterparameter-Bereichen und berechnet Durchschnitts-Strecke.
    
    Args:
        df: DataFrame mit täglichen Daten
        param: Wetterparameter-Spalte (z.B. "T2M")
        bins: Anzahl Bins (wenn bin_width None ist)
        bin_width: Schrittweite für Bins (z.B. 5.0 für 5°C-Schritte)
    
    Returns:
        DataFrame mit Spalten: bin_label, bin_center, avg_distance, std_distance, count_days
    """
    df_clean = df[[param, "daily_distance_km"]].dropna()
    
    if df_clean.empty:
        return pd.DataFrame(columns=["bin_label", "bin_center", "avg_distance", "std_distance", "count_days"])
    
    param_values = df_clean[param]
    
    # Erstelle Bins
    if bin_width is not None:
        # Feste Schrittweite
        min_val = param_values.min()
        max_val = param_values.max()
        # Runde nach unten/oben für schöne Grenzen
        min_bin = np.floor(min_val / bin_width) * bin_width
        max_bin = np.ceil(max_val / bin_width) * bin_width
        bin_edges = np.arange(min_bin, max_bin + bin_width, bin_width)
    else:
        # Automatische Bins
        bin_edges = np.linspace(param_values.min(), param_values.max(), bins + 1)
    
    # Erstelle Bin-Labels
    df_clean["bin"] = pd.cut(param_values, bins=bin_edges, include_lowest=True, duplicates="drop")
    
    # Gruppiere nach Bins
    bin_stats = df_clean.groupby("bin").agg({
        "daily_distance_km": ["mean", "std", "count"],
        param: "mean"  # Mittelwert des Parameters im Bin (für bin_center)
    }).reset_index()
    
    # Flache Spalten-Namen
    bin_stats.columns = ["bin", "avg_distance", "std_distance", "count_days", "bin_center"]
    
    # Erstelle lesbare Bin-Labels
    bin_stats["bin_label"] = bin_stats["bin"].apply(
        lambda x: f"{x.left:.1f}-{x.right:.1f}" if pd.notna(x) else "N/A"
    )
    
    # Ersetze NaN Standardabweichungen durch 0
    bin_stats["std_distance"] = bin_stats["std_distance"].fillna(0)
    
    # Sortiere nach bin_center
    bin_stats = bin_stats.sort_values("bin_center").reset_index(drop=True)
    
    return bin_stats[["bin_label", "bin_center", "avg_distance", "std_distance", "count_days"]]


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

df = load_data(DATA_PATH)

# Sidebar
st.sidebar.header("Filter & Einstellungen")

# Zeitraum-Auswahl: Nur vordefinierte Zeiträume
st.sidebar.subheader("Zeitraum")
period_name = st.sidebar.selectbox(
    "Vordefinierte Zeiträume",
    list(PREDEFINED_PERIODS.keys()),
    key="predefined_period"
)
period_info = PREDEFINED_PERIODS[period_name]
start_date = period_info["start"]
end_date = period_info["end"]
st.sidebar.caption(period_info["description"])

# Filter nach Zeitraum
df_period = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

if df_period.empty:
    st.error("Keine Daten für den gewählten Zeitraum gefunden.")
    st.stop()

# Automatisch Jahre aus dem Zeitraum extrahieren
selected_years = sorted(df_period["year"].dropna().unique())

# Automatisch alle Zebras finden, die im gewählten Zeitraum getrackt wurden
selected_zebras = sorted(df_period["individual_local_identifier"].dropna().unique())

if not selected_zebras:
    st.warning("Keine Zebras im gewählten Zeitraum gefunden.")
    st.stop()

# Gemeinsame Zebras finden (nur für Anzeige)
common_zebras_df = find_common_zebras(df_period, selected_years)

# Immer Durchschnittswerte verwenden
aggregation_mode = "Durchschnittswerte"
common_days_only = False  # Immer alle Tage verwenden

# Robust-Methode für Ausreißer-Behandlung
st.sidebar.subheader("Ausreißer-Behandlung")
robust_method = st.sidebar.selectbox(
    "Methode für Strecken-Durchschnitt",
    ["mean", "median", "trimmed", "iqr_filtered"],
    format_func=lambda x: {
        "mean": "Mittelwert (Standard)",
        "median": "Median (sehr robust)",
        "trimmed": "Getrimmter Mittelwert (oberste/unterste 10% entfernt)",
        "iqr_filtered": "IQR-gefiltert (statistische Ausreißer entfernt)"
    }[x],
    index=1,  # Standard: Median
    help="Wie sollen Ausreißer bei der Berechnung des Durchschnitts behandelt werden?",
    key="robust_method"
)

# Wetterparameter
st.sidebar.divider()
st.sidebar.subheader("Wetterparameter")
show_t2m = st.sidebar.checkbox("Temperatur Ø (T2M)", value=True, key="show_t2m")
show_t2m_max = st.sidebar.checkbox("Temperatur Max (T2M_MAX)", value=False, key="show_t2m_max")
show_precip = st.sidebar.checkbox("Niederschlag (PRECTOTCORR)", value=True, key="show_precip")
show_humidity = st.sidebar.checkbox("Luftfeuchtigkeit (RH2M)", value=False, key="show_humidity")
show_wind = st.sidebar.checkbox("Windgeschwindigkeit (WS2M)", value=False, key="show_wind")

# Visualisierungs-Optionen
st.sidebar.divider()
st.sidebar.subheader("Visualisierungs-Optionen")
line_shape = st.sidebar.radio(
    "Linien-Stil",
    ["Linear", "Glatt (Spline)"],
    index=0,
    help="Linear: Gerade Linien zwischen Punkten (zeigt Peaks)\nGlatt: Runde, flüssige Kurven ohne Peaks",
    key="line_shape"
)
show_moving_avg = st.sidebar.checkbox("7-Tage Moving Average", value=False, key="show_moving_avg")
show_year_boundaries = st.sidebar.checkbox("Jahresgrenzen markieren", value=True, key="show_year_boundaries")
show_regression_line = st.sidebar.checkbox(
    "Trend-Linie in Scatter-Plots",
    value=True,
    help="Zeigt Regressions-Linie in Scatter-Plots an",
    key="show_regression_line"
)

# Binning-Einstellungen für Trend-Analyse
st.sidebar.subheader("Trend-Analyse Einstellungen")
binning_method = st.sidebar.radio(
    "Binning-Methode",
    ["Automatisch", "Feste Schrittweite"],
    index=0,
    help="Wie sollen Temperatur-Bereiche eingeteilt werden?",
    key="binning_method"
)
if binning_method == "Feste Schrittweite":
    bin_width = st.sidebar.number_input(
        "Schrittweite (°C)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=1.0,
        help="Temperatur-Schrittweite für Binning (z.B. 5°C = 20-25°C, 25-30°C, etc.)",
        key="bin_width"
    )
    num_bins = None
else:
    num_bins = st.sidebar.slider(
        "Anzahl Bereiche",
        min_value=5,
        max_value=20,
        value=10,
        help="Anzahl Temperatur-Bereiche für Binning-Analyse",
        key="num_bins"
    )
    bin_width = None

# Filter für alle Zebras im Zeitraum
df_filtered = df_period[df_period["individual_local_identifier"].isin(selected_zebras)].copy()
df_filtered = df_filtered.dropna(subset=["location_lat", "location_long"])

if df_filtered.empty:
    st.error("Keine Daten für Zebras im gewählten Zeitraum vorhanden.")
    st.stop()

# Berechne tägliche Strecken
df_daily = calculate_daily_distances(df_filtered)

# Füge year und month Spalten hinzu für Filterung
df_daily["year"] = df_daily["date"].dt.year
df_daily["month"] = df_daily["date"].dt.month

# Metadaten-Sektion
st.subheader("📊 Zeitraum-Übersicht")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Start-Datum", start_date.strftime("%Y-%m-%d"))
with col2:
    st.metric("End-Datum", end_date.strftime("%Y-%m-%d"))
with col3:
    total_days = (end_date - start_date).days + 1
    tracked_days = df_period["date"].nunique()
    st.metric("Erfasste Tage", f"{tracked_days:,} / {total_days}")
with col4:
    coverage_pct = (tracked_days / total_days) * 100
    st.metric("Abdeckung", f"{coverage_pct:.1f}%")

st.progress(coverage_pct / 100, text=f"Datenabdeckung: {tracked_days} von {total_days} Tagen ({coverage_pct:.1f}%)")

# Jahres-Vergleichstabelle
st.subheader("📅 Jahres-Vergleich")
year_comparison = []
for year in selected_years:
    df_year = df_period[df_period["year"] == year]
    year_comparison.append({
        "Jahr": year,
        "Erster Datensatz": df_year["timestamp"].min().strftime("%Y-%m-%d"),
        "Letzter Datensatz": df_year["timestamp"].max().strftime("%Y-%m-%d"),
        "Erfasste Tage": df_year["date"].nunique(),
        "Zebras": len(df_year["individual_local_identifier"].dropna().unique())
    })

df_year_comp = pd.DataFrame(year_comparison)
st.dataframe(df_year_comp, use_container_width=True, hide_index=True)

# Zebras-Übersicht
st.subheader(f"🦓 Zebras im Zeitraum ({len(selected_zebras)} Zebras)")
zebra_info = []
for zebra_id in selected_zebras:
    df_zebra = df_period[df_period["individual_local_identifier"] == zebra_id]
    zebra_info.append({
        "Zebra-ID": zebra_id,
        "Erster Datensatz": df_zebra["timestamp"].min().strftime("%Y-%m-%d"),
        "Letzter Datensatz": df_zebra["timestamp"].max().strftime("%Y-%m-%d"),
        "Tracking-Tage": df_zebra["date"].nunique(),
        "GPS-Punkte": len(df_zebra)
    })

if zebra_info:
    df_zebra_info = pd.DataFrame(zebra_info)
    st.dataframe(df_zebra_info, use_container_width=True, hide_index=True)

# Gemeinsame Zebras-Übersicht (wenn mehrere Jahre)
if len(selected_years) > 1:
    if not common_zebras_df.empty:
        st.subheader("🦓 Gemeinsame Zebras (in mehreren Jahren)")
        st.dataframe(common_zebras_df, use_container_width=True, hide_index=True)
    else:
        st.info("Keine gemeinsamen Zebras zwischen den ausgewählten Jahren gefunden.")

st.divider()

# Jahreszeiten-Filter
st.subheader("📅 Jahreszeiten-Filter")
season_option = st.radio(
    "Wähle eine Jahreszeit für die Analyse:",
    ["Ganzes Jahr", "Trockenzeit (Mai-Oktober)", "Regenzeit (November-April)"],
    horizontal=True,
    key="season_filter"
)

# Filtere Daten nach Jahreszeit
if season_option == "Trockenzeit (Mai-Oktober)":
    df_daily = filter_by_season(df_daily, "Trockenzeit", selected_years)
    season_label = "Trockenzeit"
elif season_option == "Regenzeit (November-April)":
    df_daily = filter_by_season(df_daily, "Regenzeit", selected_years)
    season_label = "Regenzeit"
else:
    season_label = "Ganzes Jahr"

if df_daily.empty:
    st.warning(f"Keine Daten für die ausgewählte Jahreszeit ({season_label}) vorhanden.")
    st.stop()

# Aggregiere Daten: Immer Durchschnittswerte (mit robustem Mittelwert)
df_aggregated = aggregate_daily_data(
    df_daily,
    mode="average",
    zebras=selected_zebras,
    common_days_only=common_days_only,
    robust_method=robust_method
)

if df_aggregated.empty:
    st.error("Keine aggregierten Daten verfügbar.")
    st.stop()

method_label = {
    "mean": "Mittelwert",
    "median": "Median",
    "trimmed": "Getrimmter Mittelwert",
    "iqr_filtered": "IQR-gefiltert"
}[robust_method]
st.caption(f"📊 Aktuell angezeigt: {season_label} - {len(df_aggregated)} Datenpunkte - {method_label} aller Zebras")

st.divider()

# Tabs für Visualisierungen
tab1, tab2, tab3, tab4 = st.tabs(["📈 Zeitreihen", "🔍 Scatter-Plots", "📊 Vergleich", "📊 Trend-Analyse"])

# ============================================================================
# TAB 1: Zeitreihen
# ============================================================================
with tab1:
    st.subheader(f"Zeitreihen: {season_label}")
    
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
    
    # Anzahl der Subplots
    n_plots = 1 + len(weather_params)
    
    # Erstelle Subplots
    subplot_titles = ["Tägliche zurückgelegte Strecke"]
    subplot_titles.extend([f"{label}" for _, label, _ in weather_params])
    
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True,
    )
    
    # Plot Strecke: Immer Durchschnittswerte
    df_plot = df_aggregated[df_aggregated["zebra_id"] == "Durchschnitt"]
    line_shape_plotly = "spline" if line_shape == "Glatt (Spline)" else "linear"
    fig.add_trace(
        go.Scatter(
            x=df_plot["date"],
            y=df_plot["daily_distance_km"],
            mode="lines+markers",
            name="Durchschnitt",
            line=dict(color="blue", width=2, shape=line_shape_plotly),
            marker=dict(size=4, opacity=0.7),
        ),
        row=1,
        col=1,
    )
    
    # Moving Average
    if show_moving_avg:
        df_plot = df_aggregated[df_aggregated["zebra_id"] == "Durchschnitt"].sort_values("date")
        moving_avg = df_plot["daily_distance_km"].rolling(window=7, center=True).mean()
        line_shape_plotly = "spline" if line_shape == "Glatt (Spline)" else "linear"
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=moving_avg,
                mode="lines",
                name="7-Tage Durchschnitt",
                line=dict(color="darkblue", width=2, dash="dash", shape=line_shape_plotly),
            ),
            row=1,
            col=1,
        )
    
    # Wetterparameter: Immer Durchschnittswerte
    line_shape_plotly = "spline" if line_shape == "Glatt (Spline)" else "linear"
    for idx, (param, label, color) in enumerate(weather_params, start=2):
        df_plot = df_aggregated[df_aggregated["zebra_id"] == "Durchschnitt"]
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot[param],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2, shape=line_shape_plotly),
                marker=dict(size=4, opacity=0.7),
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
    
    # Jahresgrenzen markieren
    if show_year_boundaries:
        for year in selected_years:
            year_start = pd.Timestamp(year, 1, 1)
            if start_date <= year_start <= end_date:
                # Verwende add_shape statt add_vline, da add_vline Probleme mit datetime hat
                # Das Datum muss im gleichen Format wie die x-Achse sein (datetime)
                for row in range(1, n_plots + 1):
                    # Für Subplots: yref-Syntax - erster Subplot hat keine Nummer
                    if row == 1:
                        yref = "y domain"
                    else:
                        yref = f"y{row} domain"
                    
                    fig.add_shape(
                        type="line",
                        x0=year_start,
                        x1=year_start,
                        y0=0,
                        y1=1,
                        yref=yref,
                        xref="x",  # x-Achse ist shared, daher einfach "x"
                        line=dict(color="gray", width=1, dash="dash"),
                        opacity=0.5,
                        row=row,
                        col=1,
                    )
                    # Annotation separat hinzufügen
                    fig.add_annotation(
                        x=year_start,
                        y=1.02,
                        yref=yref,
                        xref="x",
                        text=f"{year}",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        row=row,
                        col=1,
                    )
    
    # Update Layout
    fig.update_layout(
        height=300 * n_plots,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig.update_yaxes(title_text="Strecke (km)", row=1, col=1)
    for idx, (_, label, _) in enumerate(weather_params, start=2):
        fig.update_yaxes(title_text=label, row=idx, col=1)
    fig.update_xaxes(title_text="Datum", row=n_plots, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: Scatter-Plots
# ============================================================================
with tab2:
    st.subheader(f"Scatter-Plots: {season_label}")
    
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
        n_params = len(scatter_params)
        n_cols = 2
        n_rows = (n_params + 1) // 2
        
        fig_scatter = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Strecke vs. {label}" for _, label, _ in scatter_params],
            vertical_spacing=0.15,
        )
        
        for idx, (param, label, base_color) in enumerate(scatter_params):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            # Immer Durchschnittswerte
            df_plot = df_aggregated[df_aggregated["zebra_id"] == "Durchschnitt"]
            fig_scatter.add_trace(
                go.Scatter(
                    x=df_plot[param],
                    y=df_plot["daily_distance_km"],
                    mode="markers",
                    name=label,
                    marker=dict(color=base_color, size=6, opacity=0.6),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            
            # Korrelation
            try:
                r, p = pearsonr(df_plot["daily_distance_km"].dropna(), df_plot[param].dropna())
                corr_text = f"r = {r:.3f}" + (f" (p = {p:.4f})" if not np.isnan(p) else "")
            except:
                corr_text = "r = N/A"
            
            # Regressions-Linie hinzufügen
            if show_regression_line:
                x_reg, y_reg = calculate_regression_line(df_plot[param], df_plot["daily_distance_km"])
                if len(x_reg) > 0:
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=x_reg,
                            y=y_reg,
                            mode="lines",
                            name="Trend-Linie",
                            line=dict(color="red", width=3, dash="dash"),
                            showlegend=(idx == 0),  # Nur für ersten Parameter
                        ),
                        row=row,
                        col=col,
                    )
            
            # Korrelations-Text hinzufügen
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
        
        fig_scatter.update_layout(height=300 * n_rows)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================================
# TAB 3: Vergleich
# ============================================================================
with tab3:
    st.subheader(f"Vergleich: {season_label}")
    
    # Jahres-Vergleich
    if len(selected_years) > 1:
        st.markdown("**Jahres-Vergleich: Durchschnittswerte**")
        
        year_stats = []
        for year in selected_years:
            df_year = df_daily[df_daily["date"].dt.year == year]
            if not df_year.empty:
                year_stats.append({
                    "Jahr": year,
                    "Ø Tägliche Strecke (km)": df_year["daily_distance_km"].mean(),
                    "Max Strecke (km)": df_year["daily_distance_km"].max(),
                    "Min Strecke (km)": df_year["daily_distance_km"].min(),
                    "Ø Temperatur (°C)": df_year["T2M"].mean() if "T2M" in df_year.columns else None,
                    "Ø Niederschlag (mm)": df_year["PRECTOTCORR"].mean() if "PRECTOTCORR" in df_year.columns else None,
                })
        
        if year_stats:
            df_year_stats = pd.DataFrame(year_stats)
            st.dataframe(df_year_stats, use_container_width=True, hide_index=True)
    
    # Zebra-Vergleich
    if len(selected_zebras) > 1:
        st.markdown("**Zebra-Vergleich: Korrelationen**")
        
        corr_labels = {
            "T2M": "Temperatur Ø",
            "T2M_MAX": "Temperatur Max",
            "PRECTOTCORR": "Niederschlag",
            "RH2M": "Luftfeuchtigkeit",
            "WS2M": "Wind",
        }
        
        comparison_corrs = []
        for zebra_id in selected_zebras:
            df_zebra = df_daily[df_daily["individual_local_identifier"] == zebra_id]
            if not df_zebra.empty:
                zebra_corrs = {"Zebra-ID": zebra_id}
                for param, label in corr_labels.items():
                    if param in df_zebra.columns:
                        try:
                            r, _ = pearsonr(
                                df_zebra["daily_distance_km"].dropna(),
                                df_zebra[param].dropna(),
                            )
                            zebra_corrs[label] = f"{r:.3f}" if not np.isnan(r) else "N/A"
                        except:
                            zebra_corrs[label] = "N/A"
                comparison_corrs.append(zebra_corrs)
        
        if comparison_corrs:
            df_corr_compare = pd.DataFrame(comparison_corrs)
            st.dataframe(df_corr_compare, use_container_width=True, hide_index=True)
            
            # Heatmap
            df_corr_plot = df_corr_compare.set_index("Zebra-ID")
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

# ============================================================================
# TAB 4: Trend-Analyse
# ============================================================================
with tab4:
    st.subheader(f"Trend-Analyse: {season_label}")
    
    st.markdown("""
    **Wie funktioniert die Trend-Analyse?**
    
    Die Daten werden nach Wetterparameter-Bereichen gruppiert (z.B. 20-25°C, 25-30°C).
    Für jeden Bereich wird die durchschnittliche zurückgelegte Strecke berechnet.
    Dies zeigt klar, ob die Strecke bei höheren/niedrigeren Wetterwerten steigt oder fällt.
    
    **Statistische Kennzahlen:**
    - **r (Korrelationskoeffizient)**: Stärke des linearen Zusammenhangs (-1 bis +1)
      - |r| < 0.2: sehr schwach, 0.2-0.5: schwach, 0.5-0.7: moderat, >0.7: stark
    - **p (p-Wert)**: Statistische Signifikanz
      - p < 0.05: signifikant (*), p < 0.01: sehr signifikant (**), p < 0.001: hoch signifikant (***)
    """)
    
    trend_params = []
    if show_t2m:
        trend_params.append(("T2M", "Temperatur Ø (°C)", "red"))
    if show_t2m_max:
        trend_params.append(("T2M_MAX", "Temperatur Max (°C)", "darkred"))
    if show_precip:
        trend_params.append(("PRECTOTCORR", "Niederschlag (mm)", "blue"))
    if show_humidity:
        trend_params.append(("RH2M", "Luftfeuchtigkeit (%)", "green"))
    if show_wind:
        trend_params.append(("WS2M", "Windgeschwindigkeit (m/s)", "orange"))
    
    if not trend_params:
        st.info("Bitte wähle mindestens einen Wetterparameter in der Sidebar aus.")
    else:
        # Binning-Analyse für jeden Parameter
        for param, label, color in trend_params:
            st.markdown(f"### {label}")
            
            # Binning-Analyse durchführen
            df_plot = df_aggregated[df_aggregated["zebra_id"] == "Durchschnitt"]
            
            if binning_method == "Feste Schrittweite":
                # Bestimme Schrittweite basierend auf Parameter-Typ
                if param in ["T2M", "T2M_MAX", "T2M_MIN"]:
                    step = bin_width  # Temperatur in °C
                elif param == "PRECTOTCORR":
                    step = bin_width * 2  # Niederschlag in mm
                elif param == "RH2M":
                    step = bin_width * 5  # Luftfeuchtigkeit in %
                elif param == "WS2M":
                    step = bin_width * 0.5  # Wind in m/s
                else:
                    step = bin_width
                
                bin_results = analyze_trend_by_bins(df_plot, param, bins=10, bin_width=step)
            else:
                bin_results = analyze_trend_by_bins(df_plot, param, bins=num_bins, bin_width=None)
            
            if bin_results.empty:
                st.warning(f"Keine Daten für {label} verfügbar.")
                continue
            
            # Visualisierung: Balkendiagramm mit Fehlerbalken
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Balkendiagramm
                fig_bar = go.Figure()
                
                fig_bar.add_trace(
                    go.Bar(
                        x=bin_results["bin_label"],
                        y=bin_results["avg_distance"],
                        error_y=dict(
                            type="data",
                            array=bin_results["std_distance"],
                            visible=True,
                            color="black",
                            thickness=1.5
                        ),
                        marker=dict(color=color, opacity=0.7),
                        text=bin_results["count_days"].astype(int),
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>"
                        + "Ø Strecke: %{y:.2f} km<br>"
                        + "Std: %{customdata:.2f} km<br>"
                        + "Tage: %{text}<extra></extra>",
                        customdata=bin_results["std_distance"],
                    )
                )
                
                fig_bar.update_layout(
                    title=f"Durchschnittliche Strecke nach {label}",
                    xaxis_title=label,
                    yaxis_title="Durchschnittliche tägliche Strecke (km)",
                    height=400,
                    showlegend=False,
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Tabelle mit Details
                st.markdown("**Durchschnittswerte pro Bereich**")
                display_table = bin_results.copy()
                display_table = display_table.rename(columns={
                    "bin_label": "Bereich",
                    "bin_center": f"{label} (Mittel)",
                    "avg_distance": "Ø Strecke (km)",
                    "std_distance": "Std (km)",
                    "count_days": "Tage"
                })
                display_table[f"{label} (Mittel)"] = display_table[f"{label} (Mittel)"].round(1)
                display_table["Ø Strecke (km)"] = display_table["Ø Strecke (km)"].round(2)
                display_table["Std (km)"] = display_table["Std (km)"].round(2)
                display_table["Tage"] = display_table["Tage"].astype(int)
                st.dataframe(display_table, use_container_width=True, hide_index=True)
            
            # Korrelation berechnen
            try:
                r, p = pearsonr(df_plot["daily_distance_km"].dropna(), df_plot[param].dropna())
                if not np.isnan(r):
                    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    corr_text = f"r = {r:.3f} (p = {p:.4f}){significance}"
                    corr_strength = "sehr schwach" if abs(r) < 0.2 else "schwach" if abs(r) < 0.5 else "moderat" if abs(r) < 0.7 else "stark"
                    corr_direction = "positiv" if r > 0 else "negativ"
                else:
                    corr_text = "r = N/A"
                    corr_strength = ""
                    corr_direction = ""
            except:
                corr_text = "r = N/A"
                corr_strength = ""
                corr_direction = ""
            
            # Trend-Interpretation
            if len(bin_results) > 1:
                # Berechne Trend (steigend/fallend)
                first_avg = bin_results.iloc[0]["avg_distance"]
                last_avg = bin_results.iloc[-1]["avg_distance"]
                trend_change = last_avg - first_avg
                trend_pct = (trend_change / first_avg * 100) if first_avg > 0 else 0
                
                if abs(trend_change) < 0.5:
                    trend_text = "Kein klarer Trend erkennbar"
                    trend_color = "gray"
                elif trend_change > 0:
                    trend_text = f"Steigender Trend: +{trend_change:.2f} km ({trend_pct:+.1f}%)"
                    trend_color = "green"
                else:
                    trend_text = f"Fallender Trend: {trend_change:.2f} km ({trend_pct:+.1f}%)"
                    trend_color = "red"
                
                # Kombinierte Anzeige: Trend + Korrelation
                st.info(f"""
                📈 **Binning-Trend:** {trend_text}
                
                📊 **Korrelation:** {corr_text}
                {f"({corr_strength} {corr_direction})" if corr_strength else ""}
                """)
            
            st.divider()

st.divider()
st.caption("💡 Tipp: Nutze die Sidebar, um Zeitraum und Wetterparameter anzupassen. Alle Analysen verwenden Durchschnittswerte aller Zebras im gewählten Zeitraum.")
