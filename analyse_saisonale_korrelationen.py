"""
Analyse: Korrelationen nach Jahreszeiten (Trockenzeit vs. Regenzeit)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

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
            row["prev_long"], row["prev_lat"],
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

def filter_by_season(df: pd.DataFrame, season: str, years: List[int]) -> pd.DataFrame:
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

# Daten laden
print("=" * 80)
print("SAISONALE KORRELATIONS-ANALYSE")
print("=" * 80)
print()

df = load_data(DATA_PATH)
start_date = pd.Timestamp(2008, 5, 1)
end_date = pd.Timestamp(2009, 4, 30)
df_period = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
selected_years = sorted(df_period["year"].dropna().unique())
selected_zebras = sorted(df_period["individual_local_identifier"].dropna().unique())

df_daily = calculate_daily_distances(df_period)
df_daily["year"] = df_daily["date"].dt.year
df_daily["month"] = df_daily["date"].dt.month

# Parameter-Liste
params = [
    ("T2M", "Temperatur Ø"),
    ("T2M_MAX", "Temperatur Max"),
    ("PRECTOTCORR", "Niederschlag"),
    ("RH2M", "Luftfeuchtigkeit"),
    ("WS2M", "Windgeschwindigkeit"),
]

# Gesamt-Analyse
print("=" * 80)
print("GESAMT (Mai 2008 - Apr 2009):")
print("=" * 80)
df_aggregated_total = aggregate_daily_data(df_daily, selected_zebras, robust_method="median")
print(f"Anzahl Tage: {len(df_aggregated_total)}")
print()

for param, label in params:
    try:
        r, p = pearsonr(df_aggregated_total["daily_distance_km"].dropna(), df_aggregated_total[param].dropna())
        if not np.isnan(r):
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            strength = "sehr schwach" if abs(r) < 0.2 else "schwach" if abs(r) < 0.5 else "moderat" if abs(r) < 0.7 else "stark"
            print(f"{label:20s}: r = {r:7.3f} (p = {p:.4f}) {significance:3s} [{strength}]")
        else:
            print(f"{label:20s}: r = N/A")
    except:
        print(f"{label:20s}: r = N/A")
print()

# Trockenzeit-Analyse
print("=" * 80)
print("TROCKENZEIT (Mai-Okt 2008):")
print("=" * 80)
df_trocken = filter_by_season(df_daily, "Trockenzeit", selected_years)
df_aggregated_trocken = aggregate_daily_data(df_trocken, selected_zebras, robust_method="median")
print(f"Anzahl Tage: {len(df_aggregated_trocken)}")
print(f"Ø Strecke: {df_aggregated_trocken['daily_distance_km'].mean():.2f} km")
print(f"Ø Temperatur: {df_aggregated_trocken['T2M'].mean():.1f} °C")
print()

for param, label in params:
    try:
        r, p = pearsonr(df_aggregated_trocken["daily_distance_km"].dropna(), df_aggregated_trocken[param].dropna())
        if not np.isnan(r):
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            strength = "sehr schwach" if abs(r) < 0.2 else "schwach" if abs(r) < 0.5 else "moderat" if abs(r) < 0.7 else "stark"
            print(f"{label:20s}: r = {r:7.3f} (p = {p:.4f}) {significance:3s} [{strength}]")
        else:
            print(f"{label:20s}: r = N/A")
    except:
        print(f"{label:20s}: r = N/A")
print()

# Regenzeit-Analyse
print("=" * 80)
print("REGENZEIT (Nov 2008 - Apr 2009):")
print("=" * 80)
df_regen = filter_by_season(df_daily, "Regenzeit", selected_years)
df_aggregated_regen = aggregate_daily_data(df_regen, selected_zebras, robust_method="median")
print(f"Anzahl Tage: {len(df_aggregated_regen)}")
print(f"Ø Strecke: {df_aggregated_regen['daily_distance_km'].mean():.2f} km")
print(f"Ø Temperatur: {df_aggregated_regen['T2M'].mean():.1f} °C")
print()

for param, label in params:
    try:
        r, p = pearsonr(df_aggregated_regen["daily_distance_km"].dropna(), df_aggregated_regen[param].dropna())
        if not np.isnan(r):
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            strength = "sehr schwach" if abs(r) < 0.2 else "schwach" if abs(r) < 0.5 else "moderat" if abs(r) < 0.7 else "stark"
            print(f"{label:20s}: r = {r:7.3f} (p = {p:.4f}) {significance:3s} [{strength}]")
        else:
            print(f"{label:20s}: r = N/A")
    except:
        print(f"{label:20s}: r = N/A")
print()

# Vergleich
print("=" * 80)
print("VERGLEICH: Trockenzeit vs. Regenzeit")
print("=" * 80)
print()
print(f"{'Parameter':<20s} {'Trockenzeit r':<15s} {'Regenzeit r':<15s} {'Unterschied':<15s}")
print("-" * 80)

for param, label in params:
    try:
        r_trocken, p_trocken = pearsonr(df_aggregated_trocken["daily_distance_km"].dropna(), df_aggregated_trocken[param].dropna())
        r_regen, p_regen = pearsonr(df_aggregated_regen["daily_distance_km"].dropna(), df_aggregated_regen[param].dropna())
        
        if not (np.isnan(r_trocken) or np.isnan(r_regen)):
            diff = abs(r_trocken) - abs(r_regen)
            diff_text = f"{diff:+.3f}"
            if abs(diff) > 0.1:
                diff_text += " (deutlich stärker in " + ("Trockenzeit" if diff > 0 else "Regenzeit") + ")"
            else:
                diff_text += " (ähnlich)"
            
            print(f"{label:20s} {r_trocken:7.3f} ({'*' if p_trocken < 0.05 else ' '})  {r_regen:7.3f} ({'*' if p_regen < 0.05 else ' '})  {diff_text}")
        else:
            print(f"{label:20s} N/A              N/A")
    except:
        print(f"{label:20s} N/A              N/A")

print()
print("=" * 80)
print("FAZIT:")
print("=" * 80)
print()
print("Wenn die Korrelationen in der Trockenzeit stärker sind, bedeutet das:")
print("- Wetter hat in der Trockenzeit einen größeren Einfluss auf die Bewegung")
print("- In der Regenzeit spielen andere Faktoren (z.B. Nahrungsverfügbarkeit)")
print("  möglicherweise eine wichtigere Rolle")
print("- Die saisonale Analyse zeigt klarere Zusammenhänge als die Gesamt-Analyse")
print()
