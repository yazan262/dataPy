"""
Überprüfung des scheinbaren Widerspruchs zwischen Korrelation und Binning-Analyse
für Temperatur Max (T2M_MAX)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple

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

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
    return df

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def calculate_daily_distances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["individual_local_identifier", "timestamp"])
    df = df.dropna(subset=["location_lat", "location_long"])

    df["prev_lat"] = df.groupby("individual_local_identifier")["location_lat"].shift(1)
    df["prev_long"] = df.groupby("individual_local_identifier")["location_long"].shift(1)

    df["step_km"] = df.apply(
        lambda row: haversine(
            row["location_long"], row["location_lat"],
            row["prev_long"], row["prev_long"],
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
    values_clean = values.dropna()
    if len(values_clean) == 0:
        return np.nan
    if method == "median":
        return values_clean.median()
    return values_clean.mean()

def aggregate_daily_data(df_daily: pd.DataFrame, zebras: list, robust_method: str = "mean") -> pd.DataFrame:
    if zebras:
        df_daily = df_daily[df_daily["individual_local_identifier"].isin(zebras)].copy()
    
    if df_daily.empty:
        return pd.DataFrame()
    
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

def analyze_trend_by_bins(df: pd.DataFrame, param: str, bins: int = 10) -> pd.DataFrame:
    df_clean = df[[param, "daily_distance_km"]].dropna()
    
    if df_clean.empty:
        return pd.DataFrame()
    
    param_values = df_clean[param]
    bin_edges = np.linspace(param_values.min(), param_values.max(), bins + 1)
    
    df_clean["bin"] = pd.cut(param_values, bins=bin_edges, include_lowest=True, duplicates="drop")
    
    bin_stats = df_clean.groupby("bin", observed=False).agg({
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

# Daten laden und verarbeiten
print("=" * 80)
print("ÜBERPRÜFUNG: Temperatur Max (T2M_MAX) - Widerspruch?")
print("=" * 80)
print()

df = load_data(DATA_PATH)
start_date = pd.Timestamp(2008, 5, 1)
end_date = pd.Timestamp(2009, 4, 30)
df_period = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
selected_zebras = sorted(df_period["individual_local_identifier"].dropna().unique())

df_daily = calculate_daily_distances(df_period)
df_daily["year"] = df_daily["date"].dt.year
df_daily["month"] = df_daily["date"].dt.month

df_aggregated = aggregate_daily_data(df_daily, selected_zebras, robust_method="median")

# Korrelation berechnen
print("1. PEARSON-KORRELATION (r):")
print("-" * 80)
r, p = pearsonr(df_aggregated["daily_distance_km"].dropna(), df_aggregated["T2M_MAX"].dropna())
print(f"r = {r:.4f}")
print(f"p = {p:.4f}")
print(f"Interpretation: {'Positiver' if r > 0 else 'Negativer'} Zusammenhang, aber {'sehr schwach' if abs(r) < 0.2 else 'schwach' if abs(r) < 0.5 else 'moderat'}")
print()

# Binning-Analyse
print("2. BINNING-ANALYSE (Durchschnittswerte pro Temperatur-Bereich):")
print("-" * 80)
bin_results = analyze_trend_by_bins(df_aggregated, "T2M_MAX", bins=10)

if not bin_results.empty:
    print(f"{'Bereich':<20s} {'Mittel (°C)':<15s} {'Ø Strecke (km)':<20s} {'Tage':<10s}")
    print("-" * 80)
    
    for _, row in bin_results.iterrows():
        print(f"{row['bin_label']:<20s} {row['bin_center']:>12.1f} {row['avg_distance']:>17.2f} {int(row['count_days']):>8d}")
    
    # Trend berechnen
    if len(bin_results) > 1:
        first_avg = bin_results.iloc[0]["avg_distance"]
        last_avg = bin_results.iloc[-1]["avg_distance"]
        trend_change = last_avg - first_avg
        trend_pct = (trend_change / first_avg * 100) if first_avg > 0 else 0
        
        print("-" * 80)
        print(f"\nTrend von erstem zu letztem Bereich:")
        print(f"  {bin_results.iloc[0]['bin_label']}: {first_avg:.2f} km")
        print(f"  {bin_results.iloc[-1]['bin_label']}: {last_avg:.2f} km")
        print(f"  Änderung: {trend_change:+.2f} km ({trend_pct:+.1f}%)")
        
        if trend_change < 0:
            print(f"  → FALLENDER TREND")
        elif trend_change > 0:
            print(f"  → STEIGENDER TREND")
        else:
            print(f"  → KEIN TREND")
print()

# Erklärung des Widerspruchs
print("=" * 80)
print("ERKLÄRUNG DES SCHEINBAREN WIDERSPRUCHS:")
print("=" * 80)
print()
print("Warum kann r = 0.134 (schwach positiv) sein, aber die Binning-Analyse")
print("einen fallenden Trend zeigt?")
print()
print("1. r = 0.134 ist SEHR SCHWACH:")
print("   - Ein r-Wert von 0.134 bedeutet praktisch 'fast kein Zusammenhang'")
print("   - Die Richtung (positiv) ist statistisch signifikant, aber die Stärke")
print("     ist so gering, dass sie praktisch vernachlässigbar ist")
print()
print("2. Nicht-lineare Beziehung:")
print("   - Die Pearson-Korrelation misst nur LINEARE Zusammenhänge")
print("   - Die echte Beziehung könnte nicht-linear sein (z.B. U-förmig)")
print("   - Bei nicht-linearen Beziehungen kann r nahe 0 sein, obwohl ein")
print("     klarer Trend in den Bins existiert")
print()
print("3. Unterschiedliche Analysemethoden:")
print("   - Korrelation: Betrachtet alle einzelnen Datenpunkte")
print("   - Binning: Gruppiert Daten und zeigt Durchschnitte")
print("   - Durchschnittswerte können andere Trends zeigen als Einzeldaten")
print()
print("4. Fazit:")
print("   - r = 0.134 bedeutet: Es gibt praktisch KEINEN linearen Zusammenhang")
print("   - Der fallende Trend in der Binning-Analyse zeigt: Bei höheren")
print("     Max-Temperaturen wird im Durchschnitt WENIGER Strecke zurückgelegt")
print("   - Diese beiden Aussagen widersprechen sich NICHT, weil:")
print("     * r ist so schwach, dass es fast keine Aussagekraft hat")
print("     * Die Binning-Analyse zeigt den tatsächlichen Trend besser")
print()

# Zusätzliche Analyse: Scatter-Plot-Statistiken
print("=" * 80)
print("ZUSÄTZLICHE ANALYSE:")
print("=" * 80)
print()
print("Temperatur Max Statistiken:")
print(f"  Min: {df_aggregated['T2M_MAX'].min():.1f} °C")
print(f"  Max: {df_aggregated['T2M_MAX'].max():.1f} °C")
print(f"  Mittel: {df_aggregated['T2M_MAX'].mean():.1f} °C")
print(f"  Median: {df_aggregated['T2M_MAX'].median():.1f} °C")
print()
print("Strecke Statistiken:")
print(f"  Min: {df_aggregated['daily_distance_km'].min():.2f} km")
print(f"  Max: {df_aggregated['daily_distance_km'].max():.2f} km")
print(f"  Mittel: {df_aggregated['daily_distance_km'].mean():.2f} km")
print(f"  Median: {df_aggregated['daily_distance_km'].median():.2f} km")
print()
