"""
Analyse: Finde den besten 12-Monats-Zeitraum für Trockenzeit + Regenzeit Analyse
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# Datenpfad
DATA_PATH = Path(__file__).parent / "data" / "zebra_weather_cleaned.csv"

print("=" * 80)
print("ANALYSE: Bester 12-Monats-Zeitraum für Trockenzeit + Regenzeit")
print("=" * 80)
print()

# Daten laden
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["date"] = df["timestamp"].dt.date

# Jahreszeiten-Definition
# Trockenzeit: Mai-Oktober (Monate 5-10)
# Regenzeit: November-April (Monate 11, 12, 1, 2, 3, 4)

print("Verfügbare Daten:")
print(f"  Start: {df['timestamp'].min()}")
print(f"  Ende: {df['timestamp'].max()}")
print()

# Mögliche 12-Monats-Zeiträume analysieren
# Ein vollständiger Zyklus: Regenzeit (Nov-Dez + Jan-Apr) + Trockenzeit (Mai-Okt)

possible_periods = []

# Option 1: Nov 2007 - Okt 2008 (Regenzeit: Nov-Dez 2007 + Jan-Apr 2008, Trockenzeit: Mai-Okt 2008)
period1_start = pd.Timestamp(2007, 11, 1)
period1_end = pd.Timestamp(2008, 10, 31)
df_period1 = df[(df["timestamp"] >= period1_start) & (df["timestamp"] <= period1_end)].copy()

if len(df_period1) > 0:
    # Regenzeit: Nov-Dez 2007 + Jan-Apr 2008
    regenzeit1 = df_period1[
        ((df_period1["year"] == 2007) & (df_period1["month"].isin([11, 12]))) |
        ((df_period1["year"] == 2008) & (df_period1["month"].isin([1, 2, 3, 4])))
    ]
    # Trockenzeit: Mai-Okt 2008
    trockenzeit1 = df_period1[df_period1["month"].isin([5, 6, 7, 8, 9, 10])]
    
    zebras_regen1 = set(regenzeit1["individual_local_identifier"].dropna().unique())
    zebras_trocken1 = set(trockenzeit1["individual_local_identifier"].dropna().unique())
    zebras_both1 = zebras_regen1 & zebras_trocken1
    
    days_regen1 = regenzeit1["date"].nunique()
    days_trocken1 = trockenzeit1["date"].nunique()
    
    possible_periods.append({
        "name": "Nov 2007 - Okt 2008",
        "start": period1_start,
        "end": period1_end,
        "zebras_regenzeit": len(zebras_regen1),
        "zebras_trockenzeit": len(zebras_trocken1),
        "zebras_beide": len(zebras_both1),
        "days_regenzeit": days_regen1,
        "days_trockenzeit": days_trocken1,
        "zebras_regenzeit_set": zebras_regen1,
        "zebras_trockenzeit_set": zebras_trocken1,
        "zebras_both_set": zebras_both1,
        "df": df_period1,
        "regenzeit_df": regenzeit1,
        "trockenzeit_df": trockenzeit1
    })

# Option 2: Nov 2008 - Apr 2009 (nur Regenzeit vollständig, Trockenzeit nur Mai-Jun 2009)
period2_start = pd.Timestamp(2008, 11, 1)
period2_end = pd.Timestamp(2009, 4, 30)
df_period2 = df[(df["timestamp"] >= period2_start) & (df["timestamp"] <= period2_end)].copy()

if len(df_period2) > 0:
    # Regenzeit: Nov-Dez 2008 + Jan-Apr 2009
    regenzeit2 = df_period2[
        ((df_period2["year"] == 2008) & (df_period2["month"].isin([11, 12]))) |
        ((df_period2["year"] == 2009) & (df_period2["month"].isin([1, 2, 3, 4])))
    ]
    
    zebras_regen2 = set(regenzeit2["individual_local_identifier"].dropna().unique())
    days_regen2 = regenzeit2["date"].nunique()
    
    possible_periods.append({
        "name": "Nov 2008 - Apr 2009 (nur Regenzeit)",
        "start": period2_start,
        "end": period2_end,
        "zebras_regenzeit": len(zebras_regen2),
        "zebras_trockenzeit": 0,
        "zebras_beide": 0,
        "days_regenzeit": days_regen2,
        "days_trockenzeit": 0,
        "zebras_regenzeit_set": zebras_regen2,
        "zebras_trockenzeit_set": set(),
        "zebras_both_set": set(),
        "df": df_period2,
        "regenzeit_df": regenzeit2,
        "trockenzeit_df": pd.DataFrame()
    })

# Option 3: Mai 2008 - Apr 2009 (Trockenzeit Mai-Okt 2008 + Regenzeit Nov-Dez 2008 + Jan-Apr 2009)
period3_start = pd.Timestamp(2008, 5, 1)
period3_end = pd.Timestamp(2009, 4, 30)
df_period3 = df[(df["timestamp"] >= period3_start) & (df["timestamp"] <= period3_end)].copy()

if len(df_period3) > 0:
    # Trockenzeit: Mai-Okt 2008
    trockenzeit3 = df_period3[(df_period3["year"] == 2008) & (df_period3["month"].isin([5, 6, 7, 8, 9, 10]))]
    # Regenzeit: Nov-Dez 2008 + Jan-Apr 2009
    regenzeit3 = df_period3[
        ((df_period3["year"] == 2008) & (df_period3["month"].isin([11, 12]))) |
        ((df_period3["year"] == 2009) & (df_period3["month"].isin([1, 2, 3, 4])))
    ]
    
    zebras_regen3 = set(regenzeit3["individual_local_identifier"].dropna().unique())
    zebras_trocken3 = set(trockenzeit3["individual_local_identifier"].dropna().unique())
    zebras_both3 = zebras_regen3 & zebras_trocken3
    
    days_regen3 = regenzeit3["date"].nunique()
    days_trocken3 = trockenzeit3["date"].nunique()
    
    possible_periods.append({
        "name": "Mai 2008 - Apr 2009",
        "start": period3_start,
        "end": period3_end,
        "zebras_regenzeit": len(zebras_regen3),
        "zebras_trockenzeit": len(zebras_trocken3),
        "zebras_beide": len(zebras_both3),
        "days_regenzeit": days_regen3,
        "days_trockenzeit": days_trocken3,
        "zebras_regenzeit_set": zebras_regen3,
        "zebras_trockenzeit_set": zebras_trocken3,
        "zebras_both_set": zebras_both3,
        "df": df_period3,
        "regenzeit_df": regenzeit3,
        "trockenzeit_df": trockenzeit3
    })

# Option 4: Nov 2008 - Jun 2009 (Regenzeit vollständig, Trockenzeit nur Mai-Jun 2009)
period4_start = pd.Timestamp(2008, 11, 1)
period4_end = pd.Timestamp(2009, 6, 1)
df_period4 = df[(df["timestamp"] >= period4_start) & (df["timestamp"] <= period4_end)].copy()

if len(df_period4) > 0:
    # Regenzeit: Nov-Dez 2008 + Jan-Apr 2009
    regenzeit4 = df_period4[
        ((df_period4["year"] == 2008) & (df_period4["month"].isin([11, 12]))) |
        ((df_period4["year"] == 2009) & (df_period4["month"].isin([1, 2, 3, 4])))
    ]
    # Trockenzeit: Mai-Jun 2009 (nur teilweise)
    trockenzeit4 = df_period4[(df_period4["year"] == 2009) & (df_period4["month"].isin([5, 6]))]
    
    zebras_regen4 = set(regenzeit4["individual_local_identifier"].dropna().unique())
    zebras_trocken4 = set(trockenzeit4["individual_local_identifier"].dropna().unique())
    zebras_both4 = zebras_regen4 & zebras_trocken4
    
    days_regen4 = regenzeit4["date"].nunique()
    days_trocken4 = trockenzeit4["date"].nunique()
    
    possible_periods.append({
        "name": "Nov 2008 - Jun 2009 (Trockenzeit teilweise)",
        "start": period4_start,
        "end": period4_end,
        "zebras_regenzeit": len(zebras_regen4),
        "zebras_trockenzeit": len(zebras_trocken4),
        "zebras_beide": len(zebras_both4),
        "days_regenzeit": days_regen4,
        "days_trockenzeit": days_trocken4,
        "zebras_regenzeit_set": zebras_regen4,
        "zebras_trockenzeit_set": zebras_trocken4,
        "zebras_both_set": zebras_both4,
        "df": df_period4,
        "regenzeit_df": regenzeit4,
        "trockenzeit_df": trockenzeit4
    })

# Ausgabe der Ergebnisse
print("=" * 80)
print("MÖGLICHE 12-MONATS-ZEITRÄUME")
print("=" * 80)
print()

for period in possible_periods:
    print(f"\n{period['name']}")
    print("-" * 80)
    print(f"Zeitraum: {period['start'].strftime('%Y-%m-%d')} bis {period['end'].strftime('%Y-%m-%d')}")
    print(f"Regenzeit:")
    print(f"  - Zebras: {period['zebras_regenzeit']}")
    print(f"  - Tracking-Tage: {period['days_regenzeit']}")
    print(f"Trockenzeit:")
    print(f"  - Zebras: {period['zebras_trockenzeit']}")
    print(f"  - Tracking-Tage: {period['days_trockenzeit']}")
    print(f"Zebras in BEIDEN Jahreszeiten: {period['zebras_beide']}")
    
    if period['zebras_beide'] > 0:
        print(f"\nZebras in beiden Jahreszeiten:")
        for zebra_id in sorted(period['zebras_both_set']):
            # Tracking-Tage für dieses Zebra
            df_zebra_regen = period['regenzeit_df'][period['regenzeit_df']['individual_local_identifier'] == zebra_id]
            df_zebra_trocken = period['trockenzeit_df'][period['trockenzeit_df']['individual_local_identifier'] == zebra_id]
            days_regen_zebra = df_zebra_regen['date'].nunique()
            days_trocken_zebra = df_zebra_trocken['date'].nunique()
            print(f"  {zebra_id}: Regenzeit {days_regen_zebra} Tage, Trockenzeit {days_trocken_zebra} Tage")

# Finde den besten Zeitraum
print("\n" + "=" * 80)
print("EMPFEHLUNG")
print("=" * 80)
print()

# Bewerte die Perioden
best_period = None
best_score = -1

for period in possible_periods:
    # Score basierend auf:
    # - Zebras in beiden Jahreszeiten (wichtigster Faktor)
    # - Vollständigkeit der Jahreszeiten (mindestens 150 Tage pro Jahreszeit)
    # - Anzahl Zebras insgesamt
    
    score = 0
    
    # Zebras in beiden Jahreszeiten (höchste Priorität)
    score += period['zebras_beide'] * 100
    
    # Vollständigkeit der Jahreszeiten
    if period['days_regenzeit'] >= 150:
        score += 50
    if period['days_trockenzeit'] >= 150:
        score += 50
    
    # Anzahl Zebras insgesamt
    score += period['zebras_regenzeit'] * 5
    score += period['zebras_trockenzeit'] * 5
    
    if score > best_score:
        best_score = score
        best_period = period

if best_period:
    print(f"✅ BESTER ZEITRAUM: {best_period['name']}")
    print(f"\nZeitraum: {best_period['start'].strftime('%Y-%m-%d')} bis {best_period['end'].strftime('%Y-%m-%d')}")
    print(f"\nRegenzeit:")
    print(f"  - Monate: Nov-Dez {best_period['start'].year} + Jan-Apr {best_period['end'].year}")
    print(f"  - Tracking-Tage: {best_period['days_regenzeit']}")
    print(f"  - Zebras: {best_period['zebras_regenzeit']}")
    print(f"\nTrockenzeit:")
    if best_period['days_trockenzeit'] > 0:
        print(f"  - Monate: Mai-Okt {best_period['start'].year if best_period['start'].month >= 5 else best_period['end'].year}")
        print(f"  - Tracking-Tage: {best_period['days_trockenzeit']}")
        print(f"  - Zebras: {best_period['zebras_trockenzeit']}")
    else:
        print(f"  - Keine vollständige Trockenzeit in diesem Zeitraum")
    
    print(f"\nZebras in BEIDEN Jahreszeiten: {best_period['zebras_beide']}")
    
    if best_period['zebras_beide'] > 0:
        print(f"\nGemeinsame Zebras (können für Multi-Jahr-Analyse verwendet werden):")
        for zebra_id in sorted(best_period['zebras_both_set']):
            df_zebra_regen = best_period['regenzeit_df'][best_period['regenzeit_df']['individual_local_identifier'] == zebra_id]
            df_zebra_trocken = best_period['trockenzeit_df'][best_period['trockenzeit_df']['individual_local_identifier'] == zebra_id]
            days_regen_zebra = df_zebra_regen['date'].nunique()
            days_trocken_zebra = df_zebra_trocken['date'].nunique()
            print(f"  {zebra_id}: Regenzeit {days_regen_zebra} Tage, Trockenzeit {days_trocken_zebra} Tage")
    
    print(f"\n📊 Für die Multi-Jahr-Analyse:")
    print(f"   - Dieser Zeitraum deckt beide Jahreszeiten ab")
    print(f"   - {best_period['zebras_beide']} Zebras können über beide Jahreszeiten analysiert werden")
    print(f"   - Ideal für Strecke-Wetter-Vergleiche zwischen Trocken- und Regenzeit")

print("\n" + "=" * 80)
