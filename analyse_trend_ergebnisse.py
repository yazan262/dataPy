"""
Analyse-Skript für Trend-Analyse: Generiert und analysiert die Ergebnisse
der Multi-Jahr Strecke-Wetter Analyse.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

# Pearson-Korrelation
try:
    from scipy.stats import pearsonr
except ImportError:
    def pearsonr(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2:
            return np.nan, np.nan
        x_clean = x[mask]
        y_clean = y[mask]
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        n = len(x_clean)
        if n < 3:
            p = np.nan
        else:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2)) if abs(corr) < 1 else np.inf
            p = 2 * (1 - abs(t_stat) / (abs(t_stat) + 1)) if np.isfinite(t_stat) else 0.0
        return corr, p

DATA_PATH = Path(__file__).resolve().parent / "data" / "zebra_weather_cleaned.csv"

# Vordefinierte Zeiträume
PREDEFINED_PERIODS = {
    "Mai 2008 - Apr 2009 (empfohlen)": {
        "start": pd.Timestamp(2008, 5, 1),
        "end": pd.Timestamp(2009, 4, 30),
        "description": "Vollständige Trockenzeit (Mai-Okt 2008) + Regenzeit (Nov-Dez 2008 + Jan-Apr 2009)"
    },
}

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

def calculate_daily_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die tägliche zurückgelegte Strecke pro Zebra"""
    df = df.copy()
    df = df.sort_values(["individual_local_identifier", "timestamp"])
    df = df.dropna(subset=["location_lat", "location_long"])

    df["prev_lat"] = df.groupby("individual_local_identifier")["location_lat"].shift(1)
    df["prev_long"] = df.groupby("individual_local_identifier")["location_long"].shift(1)

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

    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    daily_df = (
        df.groupby(["date", "individual_local_identifier"])
        .agg({
            "step_km": "sum",
            "T2M": "mean",
            "PRECTOTCORR": "mean",
            "RH2M": "mean",
            "WS2M": "mean",
            "T2M_MAX": "max",
            "T2M_MIN": "min",
        })
        .reset_index()
    )
    daily_df = daily_df.rename(columns={"step_km": "daily_distance_km"})
    return daily_df

def robust_mean(values: pd.Series, method: str = "mean") -> float:
    """Berechnet robusten Mittelwert"""
    values_clean = values.dropna()
    if len(values_clean) == 0:
        return np.nan
    if method == "median":
        return values_clean.median()
    elif method == "trimmed":
        q10 = values_clean.quantile(0.10)
        q90 = values_clean.quantile(0.90)
        trimmed = values_clean[(values_clean >= q10) & (values_clean <= q90)]
        return trimmed.mean() if len(trimmed) > 0 else values_clean.mean()
    elif method == "iqr_filtered":
        Q1 = values_clean.quantile(0.25)
        Q3 = values_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered = values_clean[(values_clean >= lower_bound) & (values_clean <= upper_bound)]
        return filtered.mean() if len(filtered) > 0 else values_clean.mean()
    else:
        return values_clean.mean()

def aggregate_daily_data(
    df_daily: pd.DataFrame,
    mode: str = "average",
    zebras: Optional[List[str]] = None,
    common_days_only: bool = False,
    robust_method: str = "mean"
) -> pd.DataFrame:
    """Aggregiert tägliche Daten nach gewähltem Modus"""
    if zebras:
        df_daily = df_daily[df_daily["individual_local_identifier"].isin(zebras)].copy()
    
    if df_daily.empty:
        return pd.DataFrame()
    
    if mode == "average":
        aggregated_list = []
        for date in df_daily["date"].unique():
            df_date = df_daily[df_daily["date"] == date]
            aggregated_list.append({
                "date": date,
                "daily_distance_km": robust_mean(df_date["daily_distance_km"], robust_method),
                "T2M": df_date["T2M"].mean(),
                "PRECTOTCORR": df_date["PRECTOTCORR"].mean(),
                "RH2M": df_date["RH2M"].mean(),
                "WS2M": df_date["WS2M"].mean(),
                "T2M_MAX": df_date["T2M_MAX"].mean(),
                "T2M_MIN": df_date["T2M_MIN"].mean(),
            })
        aggregated = pd.DataFrame(aggregated_list)
        aggregated["zebra_id"] = "Durchschnitt"
        return aggregated
    else:
        return df_daily.copy()

def calculate_regression_line(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Berechnet Regressions-Linie"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x[mask])
    y_clean = np.array(y[mask])
    
    if len(x_clean) < 2:
        return np.array([]), np.array([])
    
    coeffs = np.polyfit(x_clean, y_clean, 1)
    a, b = coeffs
    
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = a * x_line + b
    
    return x_line, y_line

def analyze_trend_by_bins(
    df: pd.DataFrame,
    param: str,
    bins: int = 10,
    bin_width: Optional[float] = None
) -> pd.DataFrame:
    """Binning-Analyse"""
    df_clean = df[[param, "daily_distance_km"]].dropna()
    
    if df_clean.empty:
        return pd.DataFrame(columns=["bin_label", "bin_center", "avg_distance", "std_distance", "count_days"])
    
    param_values = df_clean[param]
    
    if bin_width is not None:
        min_val = param_values.min()
        max_val = param_values.max()
        min_bin = np.floor(min_val / bin_width) * bin_width
        max_bin = np.ceil(max_val / bin_width) * bin_width
        bin_edges = np.arange(min_bin, max_bin + bin_width, bin_width)
    else:
        bin_edges = np.linspace(param_values.min(), param_values.max(), bins + 1)
    
    df_clean["bin"] = pd.cut(param_values, bins=bin_edges, include_lowest=True, duplicates="drop")
    
    bin_stats = df_clean.groupby("bin").agg({
        "daily_distance_km": ["mean", "std", "count"],
        param: "mean"
    }).reset_index()
    
    bin_stats.columns = ["bin", "avg_distance", "std_distance", "count_days", "bin_center"]
    
    bin_stats["bin_label"] = bin_stats["bin"].apply(
        lambda x: f"{x.left:.1f}-{x.right:.1f}" if pd.notna(x) else "N/A"
    )
    
    bin_stats["std_distance"] = bin_stats["std_distance"].fillna(0)
    bin_stats = bin_stats.sort_values("bin_center").reset_index(drop=True)
    
    return bin_stats[["bin_label", "bin_center", "avg_distance", "std_distance", "count_days"]]

def filter_by_season(df: pd.DataFrame, season: str, years: List[int]) -> pd.DataFrame:
    """Filtert Daten nach Jahreszeit"""
    if season == "Trockenzeit":
        return df[(df["month"].isin([5, 6, 7, 8, 9, 10])) & (df["year"].isin(years))].copy()
    elif season == "Regenzeit":
        df_regen = pd.DataFrame()
        for year in years:
            df_nov_dec = df[(df["year"] == year - 1) & (df["month"].isin([11, 12]))]
            df_jan_apr = df[(df["year"] == year) & (df["month"].isin([1, 2, 3, 4]))]
            df_regen = pd.concat([df_regen, df_nov_dec, df_jan_apr], ignore_index=True)
        return df_regen
    else:
        return df.copy()

# ============================================================================
# ANALYSE DURCHFÜHREN
# ============================================================================

print("=" * 80)
print("TREND-ANALYSE: Multi-Jahr Strecke-Wetter Analyse")
print("=" * 80)
print()

# Daten laden
print("📊 Lade Daten...")
df = load_data(DATA_PATH)
print(f"   Geladen: {len(df)} Datenpunkte")
print()

# Zeitraum auswählen
period_name = "Mai 2008 - Apr 2009 (empfohlen)"
period_info = PREDEFINED_PERIODS[period_name]
start_date = period_info["start"]
end_date = period_info["end"]

print(f"📅 Zeitraum: {period_name}")
print(f"   {start_date.date()} bis {end_date.date()}")
print(f"   {period_info['description']}")
print()

# Filter nach Zeitraum
df_period = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
selected_years = sorted(df_period["year"].dropna().unique())
selected_zebras = sorted(df_period["individual_local_identifier"].dropna().unique())

print(f"📈 Jahre: {selected_years}")
print(f"🦓 Zebras: {len(selected_zebras)} ({', '.join(selected_zebras[:5])}{'...' if len(selected_zebras) > 5 else ''})")
print()

# Tägliche Distanzen berechnen
print("🔄 Berechne tägliche Distanzen...")
df_daily = calculate_daily_distances(df_period)
df_daily["year"] = df_daily["date"].dt.year
df_daily["month"] = df_daily["date"].dt.month
print(f"   {len(df_daily)} Tages-Datenpunkte")
print()

# Aggregation (Durchschnittswerte mit Median)
print("📊 Aggregiere Daten (Median-Methode)...")
df_aggregated = aggregate_daily_data(df_daily, mode="average", zebras=selected_zebras, robust_method="median")
print(f"   {len(df_aggregated)} aggregierte Datenpunkte")
print()

# Statistiken
print("=" * 80)
print("GESAMT-STATISTIKEN")
print("=" * 80)
print(f"Durchschnittliche tägliche Strecke: {df_aggregated['daily_distance_km'].mean():.2f} km")
print(f"Median tägliche Strecke: {df_aggregated['daily_distance_km'].median():.2f} km")
print(f"Min/Max Strecke: {df_aggregated['daily_distance_km'].min():.2f} / {df_aggregated['daily_distance_km'].max():.2f} km")
print(f"Durchschnittliche Temperatur: {df_aggregated['T2M'].mean():.1f} °C")
print(f"Durchschnittlicher Niederschlag: {df_aggregated['PRECTOTCORR'].mean():.2f} mm")
print()

# Korrelationen
print("=" * 80)
print("KORRELATIONEN: Strecke vs. Wetterparameter")
print("=" * 80)

params = [
    ("T2M", "Temperatur Ø"),
    ("T2M_MAX", "Temperatur Max"),
    ("PRECTOTCORR", "Niederschlag"),
    ("RH2M", "Luftfeuchtigkeit"),
    ("WS2M", "Windgeschwindigkeit"),
]

for param, label in params:
    try:
        r, p = pearsonr(df_aggregated["daily_distance_km"].dropna(), df_aggregated[param].dropna())
        if not np.isnan(r):
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{label:20s}: r = {r:7.3f} (p = {p:.4f}) {significance}")
        else:
            print(f"{label:20s}: r = N/A")
    except:
        print(f"{label:20s}: r = N/A")
print()

# Regressions-Analyse
print("=" * 80)
print("REGRESSIONS-ANALYSE")
print("=" * 80)

for param, label in params:
    if param not in df_aggregated.columns:
        continue
    
    x_reg, y_reg = calculate_regression_line(df_aggregated[param], df_aggregated["daily_distance_km"])
    if len(x_reg) > 0:
        # Steigung berechnen
        slope = (y_reg[-1] - y_reg[0]) / (x_reg[-1] - x_reg[0]) if x_reg[-1] != x_reg[0] else 0
        intercept = y_reg[0] - slope * x_reg[0]
        
        print(f"\n{label}:")
        print(f"  Steigung: {slope:.4f} km pro Einheit")
        print(f"  Y-Achsenabschnitt: {intercept:.2f} km")
        print(f"  Bei {x_reg.min():.1f}: {y_reg[0]:.2f} km")
        print(f"  Bei {x_reg.max():.1f}: {y_reg[-1]:.2f} km")
        if slope > 0:
            print(f"  → Positiver Trend: Höhere {label} = Mehr Strecke")
        elif slope < 0:
            print(f"  → Negativer Trend: Höhere {label} = Weniger Strecke")
        else:
            print(f"  → Kein Trend")
print()

# Binning-Analyse
print("=" * 80)
print("BINNING-ANALYSE: Durchschnittliche Strecke nach Temperatur-Bereichen")
print("=" * 80)

bin_results = analyze_trend_by_bins(df_aggregated, "T2M", bins=10, bin_width=None)

if not bin_results.empty:
    print("\nTemperatur-Bereiche:")
    print("-" * 80)
    print(f"{'Bereich':<20s} {'Mittel (°C)':<15s} {'Ø Strecke (km)':<20s} {'Std (km)':<15s} {'Tage':<10s}")
    print("-" * 80)
    
    for _, row in bin_results.iterrows():
        print(f"{row['bin_label']:<20s} {row['bin_center']:>12.1f} {row['avg_distance']:>17.2f} {row['std_distance']:>12.2f} {int(row['count_days']):>8d}")
    
    # Trend-Interpretation
    if len(bin_results) > 1:
        first_avg = bin_results.iloc[0]["avg_distance"]
        last_avg = bin_results.iloc[-1]["avg_distance"]
        trend_change = last_avg - first_avg
        trend_pct = (trend_change / first_avg * 100) if first_avg > 0 else 0
        
        print("-" * 80)
        print(f"\nTrend-Analyse:")
        print(f"  Erster Bereich ({bin_results.iloc[0]['bin_label']}): {first_avg:.2f} km")
        print(f"  Letzter Bereich ({bin_results.iloc[-1]['bin_label']}): {last_avg:.2f} km")
        print(f"  Änderung: {trend_change:+.2f} km ({trend_pct:+.1f}%)")
        
        if abs(trend_change) < 0.5:
            print(f"  → Kein klarer Trend erkennbar")
        elif trend_change > 0:
            print(f"  → STEIGENDER TREND: Höhere Temperatur = Mehr Strecke")
        else:
            print(f"  → FALLENDER TREND: Höhere Temperatur = Weniger Strecke")
print()

# Jahreszeiten-Vergleich
print("=" * 80)
print("JAHRESZEITEN-VERGLEICH")
print("=" * 80)

for season in ["Trockenzeit", "Regenzeit"]:
    df_season = filter_by_season(df_daily, season, selected_years)
    if not df_season.empty:
        df_season_agg = aggregate_daily_data(df_season, mode="average", zebras=selected_zebras, robust_method="median")
        if not df_season_agg.empty:
            print(f"\n{season}:")
            print(f"  Durchschnittliche Strecke: {df_season_agg['daily_distance_km'].mean():.2f} km")
            print(f"  Durchschnittliche Temperatur: {df_season_agg['T2M'].mean():.1f} °C")
            print(f"  Durchschnittlicher Niederschlag: {df_season_agg['PRECTOTCORR'].mean():.2f} mm")
            print(f"  Anzahl Tage: {len(df_season_agg)}")

print()
print("=" * 80)
print("ANALYSE ABGESCHLOSSEN")
print("=" * 80)
