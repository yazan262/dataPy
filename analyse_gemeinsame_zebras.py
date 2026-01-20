"""
Analyse: Gemeinsame Zebras zwischen Jahren
Prüft, ob es Zebras gibt, die in mehreren Jahren getrackt wurden,
getrennt nach Trockenzeit und Regenzeit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Datenpfad
DATA_PATH = Path(__file__).parent / "data" / "zebra_weather_cleaned.csv"

print("=" * 80)
print("ANALYSE: Gemeinsame Zebras zwischen Jahren")
print("=" * 80)
print()

# Daten laden
print("📂 Lade Daten...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"✓ {len(df):,} Datensätze geladen")
print(f"  Zeitraum: {df['timestamp'].min()} bis {df['timestamp'].max()}")
print()

# Verfügbare Jahre
years = sorted(df["timestamp"].dt.year.dropna().unique())
print(f"📅 Verfügbare Jahre: {years}")
print()

# Jahreszeiten-Definition
# Trockenzeit: Mai-Oktober (Monate 5-10)
# Regenzeit: November-April (Monate 11, 12, 1, 2, 3, 4)

def get_season(month):
    """Bestimmt Jahreszeit basierend auf Monat"""
    if 5 <= month <= 10:
        return "Trockenzeit"
    else:  # 11, 12, 1, 2, 3, 4
        return "Regenzeit"

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["season"] = df["month"].apply(get_season)

# ============================================================================
# 1. Übersicht: Zebras pro Jahr
# ============================================================================
print("=" * 80)
print("1. ÜBERSICHT: Zebras pro Jahr")
print("=" * 80)
print()

zebras_by_year = {}
for year in years:
    df_year = df[df["year"] == year]
    zebras_year = set(df_year["individual_local_identifier"].dropna().unique())
    zebras_by_year[year] = zebras_year
    print(f"Jahr {year}:")
    print(f"  - Anzahl Zebras: {len(zebras_year)}")
    print(f"  - Erster Datensatz: {df_year['timestamp'].min()}")
    print(f"  - Letzter Datensatz: {df_year['timestamp'].max()}")
    print(f"  - Erfasste Tage: {df_year['timestamp'].dt.date.nunique()}")
    print()

# ============================================================================
# 2. Gemeinsame Zebras (gesamt, alle Jahre)
# ============================================================================
print("=" * 80)
print("2. GEMEINSAME ZEBRAS (Gesamt, alle Jahreszeiten)")
print("=" * 80)
print()

# Finde Zebras, die in mehreren Jahren vorkommen
zebra_year_matrix = defaultdict(set)
for year in years:
    for zebra_id in zebras_by_year[year]:
        zebra_year_matrix[zebra_id].add(year)

# Kategorisiere Zebras
zebras_in_multiple_years = {
    zebra_id: years_set
    for zebra_id, years_set in zebra_year_matrix.items()
    if len(years_set) > 1
}

zebras_in_single_year = {
    zebra_id: years_set
    for zebra_id, years_set in zebra_year_matrix.items()
    if len(years_set) == 1
}

print(f"Zebras in mehreren Jahren: {len(zebras_in_multiple_years)}")
print(f"Zebras nur in einem Jahr: {len(zebras_in_single_year)}")
print()

if zebras_in_multiple_years:
    print("Gemeinsame Zebras (in mehreren Jahren):")
    print("-" * 80)
    for zebra_id, years_set in sorted(zebras_in_multiple_years.items()):
        years_list = sorted(years_set)
        print(f"  {zebra_id}: Jahre {years_list}")
    print()
else:
    print("⚠️  KEINE gemeinsamen Zebras gefunden!")
    print()

# ============================================================================
# 3. Analyse nach Jahreszeiten
# ============================================================================
print("=" * 80)
print("3. ANALYSE NACH JAHRESZEITEN")
print("=" * 80)
print()

# Für jede Jahreszeit separat analysieren
for season_name in ["Trockenzeit", "Regenzeit"]:
    print(f"\n{'='*80}")
    print(f"{season_name.upper()}")
    print(f"{'='*80}")
    print()
    
    # Filtere nach Jahreszeit
    if season_name == "Trockenzeit":
        # Trockenzeit: Mai-Oktober (Monate 5-10)
        df_season = df[df["month"].isin([5, 6, 7, 8, 9, 10])].copy()
    else:  # Regenzeit
        # Regenzeit: November-April (Monate 11, 12, 1, 2, 3, 4)
        df_season = df[df["month"].isin([11, 12, 1, 2, 3, 4])].copy()
    
    # Zebras pro Jahr in dieser Jahreszeit
    zebras_by_year_season = {}
    for year in years:
        df_year_season = df_season[df_season["year"] == year]
        zebras_year_season = set(df_year_season["individual_local_identifier"].dropna().unique())
        zebras_by_year_season[year] = zebras_year_season
        
        # Tracking-Tage für diese Jahreszeit
        tracking_days = df_year_season["timestamp"].dt.date.nunique()
        
        print(f"Jahr {year} ({season_name}):")
        print(f"  - Anzahl Zebras: {len(zebras_year_season)}")
        print(f"  - Tracking-Tage: {tracking_days}")
        if len(zebras_year_season) > 0:
            date_range = f"{df_year_season['timestamp'].min().strftime('%Y-%m-%d')} bis {df_year_season['timestamp'].max().strftime('%Y-%m-%d')}"
            print(f"  - Datumsbereich: {date_range}")
        print()
    
    # Finde gemeinsame Zebras für diese Jahreszeit
    zebra_year_matrix_season = defaultdict(set)
    for year in years:
        for zebra_id in zebras_by_year_season[year]:
            zebra_year_matrix_season[zebra_id].add(year)
    
    zebras_in_multiple_years_season = {
        zebra_id: years_set
        for zebra_id, years_set in zebra_year_matrix_season.items()
        if len(years_set) > 1
    }
    
    print(f"Gemeinsame Zebras in {season_name} (mehrere Jahre): {len(zebras_in_multiple_years_season)}")
    
    if zebras_in_multiple_years_season:
        print(f"\nGemeinsame Zebras ({season_name}):")
        print("-" * 80)
        for zebra_id, years_set in sorted(zebras_in_multiple_years_season.items()):
            years_list = sorted(years_set)
            # Berechne Tracking-Tage pro Jahr für dieses Zebra
            tracking_info = []
            for year in years_list:
                df_zebra_year = df_season[
                    (df_season["individual_local_identifier"] == zebra_id) &
                    (df_season["year"] == year)
                ]
                days = df_zebra_year["timestamp"].dt.date.nunique()
                tracking_info.append(f"{year}: {days} Tage")
            
            print(f"  {zebra_id}: Jahre {years_list}")
            print(f"    Tracking: {', '.join(tracking_info)}")
        print()
    else:
        print(f"⚠️  KEINE gemeinsamen Zebras in {season_name} gefunden!")
        print()

# ============================================================================
# 4. Detaillierte Analyse: Kombination Trockenzeit + Regenzeit (12 Monate)
# ============================================================================
print("=" * 80)
print("4. KOMBINIERTE ANALYSE: Trockenzeit + Regenzeit (12 Monate)")
print("=" * 80)
print()
print("Analysiert Zebras, die sowohl in Trockenzeit als auch Regenzeit")
print("getrackt wurden, über mehrere Jahre hinweg.")
print()

# Kombiniere beide Jahreszeiten (12 Monate)
df_combined = df.copy()  # Alle Daten

zebras_by_year_combined = {}
for year in years:
    df_year = df_combined[df_combined["year"] == year]
    zebras_year = set(df_year["individual_local_identifier"].dropna().unique())
    zebras_by_year_combined[year] = zebras_year
    
    # Tracking-Tage
    tracking_days = df_year["timestamp"].dt.date.nunique()
    
    print(f"Jahr {year} (Trockenzeit + Regenzeit):")
    print(f"  - Anzahl Zebras: {len(zebras_year)}")
    print(f"  - Tracking-Tage: {tracking_days}")
    print()

# Finde gemeinsame Zebras
zebra_year_matrix_combined = defaultdict(set)
for year in years:
    for zebra_id in zebras_by_year_combined[year]:
        zebra_year_matrix_combined[zebra_id].add(year)

zebras_in_multiple_years_combined = {
    zebra_id: years_set
    for zebra_id, years_set in zebra_year_matrix_combined.items()
    if len(years_set) > 1
}

print(f"Gemeinsame Zebras (mehrere Jahre, beide Jahreszeiten): {len(zebras_in_multiple_years_combined)}")
print()

if zebras_in_multiple_years_combined:
    print("Gemeinsame Zebras (kombiniert):")
    print("-" * 80)
    
    # Detaillierte Informationen für jedes gemeinsame Zebra
    detailed_info = []
    for zebra_id, years_set in sorted(zebras_in_multiple_years_combined.items()):
        years_list = sorted(years_set)
        
        # Für jedes Jahr: Tracking-Tage in Trockenzeit und Regenzeit
        year_info = {}
        for year in years_list:
            df_zebra_year = df_combined[
                (df_combined["individual_local_identifier"] == zebra_id) &
                (df_combined["year"] == year)
            ]
            
            # Trockenzeit
            df_dry = df_zebra_year[df_zebra_year["month"].isin([5, 6, 7, 8, 9, 10])]
            days_dry = df_dry["timestamp"].dt.date.nunique() if len(df_dry) > 0 else 0
            
            # Regenzeit
            df_wet = df_zebra_year[df_zebra_year["month"].isin([11, 12, 1, 2, 3, 4])]
            days_wet = df_wet["timestamp"].dt.date.nunique() if len(df_wet) > 0 else 0
            
            # Gesamt
            days_total = df_zebra_year["timestamp"].dt.date.nunique()
            
            year_info[year] = {
                "total": days_total,
                "dry": days_dry,
                "wet": days_wet
            }
        
        detailed_info.append({
            "zebra_id": zebra_id,
            "years": years_list,
            "year_info": year_info
        })
        
        # Ausgabe
        print(f"\n{zebra_id}: Jahre {years_list}")
        for year in years_list:
            info = year_info[year]
            print(f"  {year}: {info['total']} Tage gesamt "
                  f"(Trockenzeit: {info['dry']} Tage, Regenzeit: {info['wet']} Tage)")
    
    print()
    
    # Zusammenfassung
    print("-" * 80)
    print("ZUSAMMENFASSUNG:")
    print("-" * 80)
    
    # Paare von Jahren analysieren
    year_pairs = []
    if 2007 in years and 2008 in years:
        year_pairs.append((2007, 2008))
    if 2008 in years and 2009 in years:
        year_pairs.append((2008, 2009))
    if 2007 in years and 2009 in years:
        year_pairs.append((2007, 2009))
    
    for year1, year2 in year_pairs:
        common_zebras = [
            zebra_id for zebra_id, years_set in zebras_in_multiple_years_combined.items()
            if year1 in years_set and year2 in years_set
        ]
        print(f"\nGemeinsame Zebras {year1}-{year2}: {len(common_zebras)}")
        if common_zebras:
            print(f"  IDs: {', '.join(sorted(common_zebras))}")
    
    # Alle drei Jahre
    if all(y in years for y in [2007, 2008, 2009]):
        common_all_three = [
            zebra_id for zebra_id, years_set in zebras_in_multiple_years_combined.items()
            if len(years_set) == 3 and all(y in years_set for y in [2007, 2008, 2009])
        ]
        print(f"\nGemeinsame Zebras in ALLEN drei Jahren (2007, 2008, 2009): {len(common_all_three)}")
        if common_all_three:
            print(f"  IDs: {', '.join(sorted(common_all_three))}")
    
    print()
else:
    print("⚠️  KEINE gemeinsamen Zebras gefunden!")
    print()

# ============================================================================
# 5. Empfehlung für die Multi-Jahr-Analyse
# ============================================================================
print("=" * 80)
print("5. EMPFEHLUNG FÜR MULTI-JAHR-ANALYSE")
print("=" * 80)
print()

if zebras_in_multiple_years_combined:
    print("✅ Es gibt gemeinsame Zebras zwischen den Jahren!")
    print()
    print("Empfehlungen:")
    print("  1. Die Multi-Jahr-Analyse-Seite kann erstellt werden")
    print("  2. Fokus auf Zebras, die in mehreren Jahren getrackt wurden")
    print("  3. Option 'Nur gemeinsame Zebras' sollte implementiert werden")
    print("  4. Für Vergleiche: Kombination aus Trockenzeit + Regenzeit nutzen")
    print()
    
    # Zeige die besten Kandidaten
    print("Beste Kandidaten für Multi-Jahr-Analyse:")
    print("-" * 80)
    
    # Sortiere nach Anzahl Jahre und Tracking-Tagen
    candidates = []
    for zebra_id, years_set in zebras_in_multiple_years_combined.items():
        total_days = 0
        for year in years_set:
            df_zebra_year = df_combined[
                (df_combined["individual_local_identifier"] == zebra_id) &
                (df_combined["year"] == year)
            ]
            total_days += df_zebra_year["timestamp"].dt.date.nunique()
        
        candidates.append({
            "zebra_id": zebra_id,
            "years": sorted(years_set),
            "num_years": len(years_set),
            "total_days": total_days
        })
    
    # Sortiere nach Anzahl Jahre (absteigend), dann nach Tracking-Tagen
    candidates.sort(key=lambda x: (x["num_years"], x["total_days"]), reverse=True)
    
    for i, candidate in enumerate(candidates[:10], 1):  # Top 10
        print(f"{i}. {candidate['zebra_id']}: {candidate['num_years']} Jahre, "
              f"{candidate['total_days']} Tracking-Tage gesamt")
        print(f"   Jahre: {candidate['years']}")
    
    print()
else:
    print("⚠️  KEINE gemeinsamen Zebras gefunden!")
    print()
    print("Empfehlung:")
    print("  - Die Multi-Jahr-Analyse-Seite kann trotzdem erstellt werden")
    print("  - Aber es werden keine direkten Vergleiche zwischen Jahren möglich sein")
    print("  - Fokus auf jahresübergreifende Trends und Muster")
    print()

print("=" * 80)
print("ANALYSE ABGESCHLOSSEN")
print("=" * 80)
