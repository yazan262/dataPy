"""
Analyse: Überlappung der Tracking-Tage zwischen Zebras
Um zu verstehen, wie wir mit mehreren Zebras pro Tag umgehen sollten
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Datenpfad
DATA_PATH = Path(__file__).parent / "data" / "zebra_weather_cleaned.csv"

print("=" * 80)
print("ANALYSE: Tracking-Tage Überlappung zwischen Zebras")
print("=" * 80)
print()

# Daten laden
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month

# Jahreszeiten-Definition
def get_season(month):
    if 5 <= month <= 10:
        return "Trockenzeit"
    else:
        return "Regenzeit"

df["season"] = df["month"].apply(get_season)

# Fokus auf den besten Zeitraum: Mai 2008 - Apr 2009
start_date = pd.Timestamp(2008, 5, 1)
end_date = pd.Timestamp(2009, 4, 30)
df_period = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

print(f"Zeitraum: {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}")
print()

# Trockenzeit: Mai-Okt 2008
df_dry = df_period[
    (df_period["year"] == 2008) & (df_period["month"].isin([5, 6, 7, 8, 9, 10]))
].copy()

# Regenzeit: Nov-Dez 2008 + Jan-Apr 2009
df_wet = df_period[
    ((df_period["year"] == 2008) & (df_period["month"].isin([11, 12]))) |
    ((df_period["year"] == 2009) & (df_period["month"].isin([1, 2, 3, 4])))
].copy()

# Gemeinsame Zebras finden
zebras_dry = set(df_dry["individual_local_identifier"].dropna().unique())
zebras_wet = set(df_wet["individual_local_identifier"].dropna().unique())
zebras_both = zebras_dry & zebras_wet

print(f"Zebras in Trockenzeit: {len(zebras_dry)}")
print(f"Zebras in Regenzeit: {len(zebras_wet)}")
print(f"Zebras in BEIDEN: {len(zebras_both)}")
print(f"  IDs: {sorted(zebras_both)}")
print()

# ============================================================================
# Analyse 1: Wie viele Zebras pro Tag?
# ============================================================================
print("=" * 80)
print("1. WIE VIELE ZEBRAS PRO TAG?")
print("=" * 80)
print()

def analyze_zebras_per_day(df_season, season_name):
    """Analysiert, wie viele Zebras pro Tag getrackt wurden"""
    zebras_per_day = df_season.groupby("date")["individual_local_identifier"].nunique()
    
    print(f"\n{season_name}:")
    print("-" * 80)
    print(f"Gesamtanzahl Tage: {len(zebras_per_day)}")
    print(f"\nVerteilung der Anzahl Zebras pro Tag:")
    
    zebra_count_dist = zebras_per_day.value_counts().sort_index()
    for count, days in zebra_count_dist.items():
        pct = (days / len(zebras_per_day)) * 100
        print(f"  {count} Zebra(s): {days} Tage ({pct:.1f}%)")
    
    # Tage mit nur einem Zebra
    days_one_zebra = (zebras_per_day == 1).sum()
    days_multiple_zebras = (zebras_per_day > 1).sum()
    
    print(f"\nTage mit nur 1 Zebra: {days_one_zebra} ({days_one_zebra/len(zebras_per_day)*100:.1f}%)")
    print(f"Tage mit mehreren Zebras: {days_multiple_zebras} ({days_multiple_zebras/len(zebras_per_day)*100:.1f}%)")
    
    return zebras_per_day, days_one_zebra, days_multiple_zebras

zebras_per_day_dry, days_one_dry, days_multiple_dry = analyze_zebras_per_day(df_dry, "TROCKENZEIT")
zebras_per_day_wet, days_one_wet, days_multiple_wet = analyze_zebras_per_day(df_wet, "REGENZEIT")

# ============================================================================
# Analyse 2: Welche Zebras sind an denselben Tagen getrackt?
# ============================================================================
print("\n" + "=" * 80)
print("2. ÜBERLAPPUNG: Welche Zebras sind an denselben Tagen getrackt?")
print("=" * 80)
print()

def analyze_overlap(df_season, season_name, zebras_list):
    """Analysiert die Überlappung der Tracking-Tage zwischen Zebras"""
    print(f"\n{season_name}:")
    print("-" * 80)
    
    # Für jedes Zebra: Tracking-Tage
    zebra_days = {}
    for zebra_id in zebras_list:
        df_zebra = df_season[df_season["individual_local_identifier"] == zebra_id]
        days = set(df_zebra["date"].unique())
        zebra_days[zebra_id] = days
        print(f"{zebra_id}: {len(days)} Tracking-Tage")
    
    # Überlappung zwischen allen Paaren
    print(f"\nÜberlappung zwischen Zebras (gemeinsame Tage):")
    zebra_pairs = []
    for i, zebra1 in enumerate(zebras_list):
        for zebra2 in zebras_list[i+1:]:
            common_days = zebra_days[zebra1] & zebra_days[zebra2]
            overlap_pct1 = (len(common_days) / len(zebra_days[zebra1])) * 100 if len(zebra_days[zebra1]) > 0 else 0
            overlap_pct2 = (len(common_days) / len(zebra_days[zebra2])) * 100 if len(zebra_days[zebra2]) > 0 else 0
            
            zebra_pairs.append({
                "zebra1": zebra1,
                "zebra2": zebra2,
                "common_days": len(common_days),
                "overlap_pct1": overlap_pct1,
                "overlap_pct2": overlap_pct2
            })
            
            print(f"  {zebra1} & {zebra2}: {len(common_days)} gemeinsame Tage "
                  f"({overlap_pct1:.1f}% von {zebra1}, {overlap_pct2:.1f}% von {zebra2})")
    
    return zebra_days, zebra_pairs

zebra_days_dry, pairs_dry = analyze_overlap(df_dry, "TROCKENZEIT", sorted(zebras_both))
zebra_days_wet, pairs_wet = analyze_overlap(df_wet, "REGENZEIT", sorted(zebras_both))

# ============================================================================
# Analyse 3: Beispiel-Tage analysieren
# ============================================================================
print("\n" + "=" * 80)
print("3. BEISPIEL-TAGE: Welche Zebras sind an bestimmten Tagen getrackt?")
print("=" * 80)
print()

def show_example_days(df_season, season_name, num_examples=5):
    """Zeigt Beispiel-Tage mit verschiedenen Zebra-Anzahlen"""
    zebras_per_day = df_season.groupby("date")["individual_local_identifier"].nunique()
    
    # Finde Tage mit verschiedenen Zebra-Anzahlen
    examples = {}
    for count in sorted(zebras_per_day.unique()):
        days_with_count = zebras_per_day[zebras_per_day == count].index[:num_examples]
        examples[count] = days_with_count
    
    print(f"\n{season_name}:")
    print("-" * 80)
    
    for count, days in examples.items():
        print(f"\nTage mit {count} Zebra(s) (Beispiele):")
        for day in days[:3]:  # Zeige nur 3 Beispiele
            zebras_on_day = df_season[df_season["date"] == day]["individual_local_identifier"].unique()
            print(f"  {day}: {', '.join(sorted(zebras_on_day))}")

show_example_days(df_dry, "TROCKENZEIT")
show_example_days(df_wet, "REGENZEIT")

# ============================================================================
# Analyse 4: Durchschnittswerte berechnen (Test)
# ============================================================================
print("\n" + "=" * 80)
print("4. TEST: Durchschnittswerte pro Tag (wenn mehrere Zebras)")
print("=" * 80)
print()

# Berechne tägliche Strecken (vereinfacht - nur für Demo)
def calculate_daily_distances_simple(df_season):
    """Vereinfachte Berechnung der täglichen Strecken"""
    df_sorted = df_season.sort_values(["individual_local_identifier", "timestamp"])
    
    # Berechne Schritt-Distanzen (vereinfacht)
    df_sorted["prev_lat"] = df_sorted.groupby("individual_local_identifier")["location_lat"].shift(1)
    df_sorted["prev_long"] = df_sorted.groupby("individual_local_identifier")["location_long"].shift(1)
    
    def haversine_simple(row):
        if pd.isna(row["prev_lat"]) or pd.isna(row["prev_long"]):
            return 0.0
        # Vereinfachte Distanzberechnung
        from math import radians, sin, cos, sqrt, atan2
        R = 6371  # Erdradius in km
        lat1, lon1 = radians(row["location_lat"]), radians(row["location_long"])
        lat2, lon2 = radians(row["prev_lat"]), radians(row["prev_long"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    df_sorted["step_km"] = df_sorted.apply(haversine_simple, axis=1)
    
    # Tägliche Aggregation pro Zebra
    daily_df = df_sorted.groupby(["date", "individual_local_identifier"]).agg({
        "step_km": "sum",
        "T2M": "mean",
        "PRECTOTCORR": "mean",
        "RH2M": "mean",
        "WS2M": "mean"
    }).reset_index()
    
    daily_df = daily_df.rename(columns={"step_km": "daily_distance_km"})
    
    return daily_df

print("Berechne tägliche Strecken...")
daily_dry = calculate_daily_distances_simple(df_dry)
daily_wet = calculate_daily_distances_simple(df_wet)

# Test: Durchschnittswerte pro Tag
print("\nTROCKENZEIT - Vergleich:")
print("-" * 80)
print("Option A: Einzelne Zebras (Beispiel)")
print("Option B: Durchschnittswerte pro Tag (wenn mehrere Zebras)")

# Beispiel: Ein Tag mit mehreren Zebras
example_date = daily_dry["date"].iloc[0]
zebras_on_day = daily_dry[daily_dry["date"] == example_date]

if len(zebras_on_day) > 1:
    print(f"\nBeispiel-Tag: {example_date}")
    print(f"Zebras an diesem Tag: {len(zebras_on_day)}")
    print("\nEinzelne Werte:")
    for _, row in zebras_on_day.iterrows():
        print(f"  {row['individual_local_identifier']}: {row['daily_distance_km']:.2f} km, "
              f"T2M: {row['T2M']:.2f}°C")
    
    print("\nDurchschnittswerte:")
    avg_distance = zebras_on_day["daily_distance_km"].mean()
    avg_t2m = zebras_on_day["T2M"].mean()
    print(f"  Durchschnitt Strecke: {avg_distance:.2f} km")
    print(f"  Durchschnitt Temperatur: {avg_t2m:.2f}°C")

# ============================================================================
# Analyse 5: Empfehlungen
# ============================================================================
print("\n" + "=" * 80)
print("5. EMPFEHLUNGEN FÜR DIE MULTI-JAHR-ANALYSE")
print("=" * 80)
print()

print("PROBLEM:")
print("  - Nicht alle Zebras sind an jedem Tag getrackt")
print("  - Scatter-Plots werden ungenau, wenn Datenpunkte fehlen")
print("  - Verschiedene Zebras haben unterschiedliche Tracking-Zeiträume")
print()

print("MÖGLICHE LÖSUNGEN:")
print()

print("OPTION 1: Ein Zebra pro Jahreszeit")
print("  ✅ Vorteile:")
print("     - Klare, konsistente Daten")
print("     - Keine fehlenden Werte")
print("     - Einfache Interpretation")
print("  ❌ Nachteile:")
print("     - Verliert Daten von anderen Zebras")
print("     - Weniger robust (ein Zebra könnte Ausreißer sein)")
print()

print("OPTION 2: Durchschnittswerte pro Tag")
print("  ✅ Vorteile:")
print("     - Nutzt alle verfügbaren Daten")
print("     - Reduziert Ausreißer-Effekte")
print("     - Repräsentativer für die Gruppe")
print("  ❌ Nachteile:")
print("     - Verliert individuelle Variationen")
print("     - Tage mit nur einem Zebra haben gleiches Gewicht wie Tage mit mehreren")
print()

print("OPTION 3: Nur Tage mit allen Zebras")
print("  ✅ Vorteile:")
print("     - Konsistente Daten")
print("     - Fairer Vergleich")
print("  ❌ Nachteile:")
print("     - Sehr wenige Datenpunkte")
print("     - Verliert viele Informationen")
print()

print("OPTION 4: Gewichteter Durchschnitt (nach Anzahl Zebras)")
print("  ✅ Vorteile:")
print("     - Nutzt alle Daten")
print("     - Berücksichtigt Datenqualität")
print("  ❌ Nachteile:")
print("     - Komplexer")
print()

print("OPTION 5: Flexible Auswahl (Nutzer wählt)")
print("  ✅ Vorteile:")
print("     - Maximale Flexibilität")
print("     - Nutzer kann je nach Frage wählen")
print("  ❌ Nachteile:")
print("     - Komplexere UI")
print()

print("=" * 80)
print("EMPFEHLUNG:")
print("=" * 80)
print()
print("Kombination aus Option 2 + Option 5:")
print("  - Standard: Durchschnittswerte pro Tag (nutzt alle Daten)")
print("  - Option: Einzelne Zebras auswählbar")
print("  - Option: Nur gemeinsame Tage (wenn alle Zebras Daten haben)")
print()
print("Für Scatter-Plots:")
print("  - Standard: Durchschnittswerte pro Tag")
print("  - Optional: Einzelne Zebras separat anzeigen (mit Farbcodierung)")
print()
