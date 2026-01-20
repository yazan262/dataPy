from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zebra Migration", layout="wide")
st.title("Zebra Migration & Wetter")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "zebra_weather_cleaned.csv"

PREDEFINED_PERIODS = {
    "Nov 2007 – Okt 2008 (Saison 2007–2008)": {
        "start": pd.Timestamp(2007, 11, 1),
        "end": pd.Timestamp(2008, 10, 31),
        "description": "Regenzeit: Nov–Dez 2007 + Jan–Apr 2008 | Trockenzeit: Mai–Okt 2008",
    },
    "Mai 2008 – Apr 2009 (Saison 2008–2009)": {
        "start": pd.Timestamp(2008, 5, 1),
        "end": pd.Timestamp(2009, 4, 30),
        "description": "Trockenzeit: Mai–Okt 2008 | Regenzeit: Nov–Dez 2008 + Jan–Apr 2009",
    },
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data(DATA_PATH)

# Sidebar Filter
st.sidebar.header("Filter")

st.sidebar.subheader("Zeitraum (Saison-Fenster)")
period_name = st.sidebar.selectbox(
    "Vordefinierte Zeitfenster",
    list(PREDEFINED_PERIODS.keys()),
    index=1,
    key="map_period_select",
)
period_info = PREDEFINED_PERIODS[period_name]
start_date = period_info["start"]
end_date = period_info["end"]
st.sidebar.caption(period_info["description"])

st.sidebar.subheader("Jahreszeit")
season_option = st.sidebar.radio(
    "Wähle Filter:",
    ["Ganzes Jahr", "Trockenzeit", "Regenzeit"],
    horizontal=True,
    key="map_season_filter",
    help="Trockenzeit = Mai–Okt | Regenzeit = Nov–Apr (über Jahresgrenze hinweg)",
)

# Filter nach Zeitraum (alle Zebras automatisch)
df_f = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
df_f = df_f.dropna(subset=["location_lat", "location_long"])
df_f = df_f.sort_values("timestamp")

if df_f.empty:
    st.error("Keine Daten für den gewählten Zeitraum gefunden.")
    st.stop()

# Saison-Spalte (Monats-basiert; innerhalb des gewählten Fensters ist das korrekt)
df_f["season"] = df_f["timestamp"].dt.month.map(
    lambda m: "Trockenzeit" if 5 <= m <= 10 else "Regenzeit"
)

if season_option == "Trockenzeit":
    df_f = df_f[df_f["timestamp"].dt.month.isin([5, 6, 7, 8, 9, 10])].copy()
elif season_option == "Regenzeit":
    df_f = df_f[df_f["timestamp"].dt.month.isin([11, 12, 1, 2, 3, 4])].copy()

if df_f.empty:
    st.warning("Keine Daten für die gewählte Jahreszeit im ausgewählten Zeitraum.")
    st.stop()

# Tracking-Informationen pro Zebra
st.subheader("📊 Übersicht")

zebra_count = df_f["individual_local_identifier"].dropna().nunique()
date_count = df_f["timestamp"].dt.date.nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Zeitraum", f"{start_date.date()} → {end_date.date()}")
c2.metric("Jahreszeit-Filter", season_option)
c3.metric("Zebras", f"{zebra_count:,}")
c4.metric("Tage (unique)", f"{date_count:,}")

st.divider()

# Überblick
c5, c6, c7 = st.columns(3)
c5.metric("GPS-Punkte", f"{len(df_f):,}")
c6.metric("Regen (PRECTOTCORR) Ø", f"{df_f['PRECTOTCORR'].mean():.2f}")
c7.metric("Temp (T2M) Ø", f"{df_f['T2M'].mean():.2f}" if "T2M" in df_f.columns else "N/A")

with st.expander("🦓 Zebra-Details anzeigen"):
    zebra_meta = (
        df_f.groupby("individual_local_identifier")["timestamp"]
        .agg(["min", "max", lambda s: s.dt.date.nunique(), "count"])
        .reset_index()
        .rename(
            columns={
                "individual_local_identifier": "Zebra-ID",
                "min": "Erster Datensatz",
                "max": "Letzter Datensatz",
                "<lambda_0>": "Tracking-Tage",
                "count": "GPS-Punkte",
            }
        )
        .sort_values("Tracking-Tage", ascending=False)
    )
    zebra_meta["Erster Datensatz"] = zebra_meta["Erster Datensatz"].dt.strftime("%Y-%m-%d")
    zebra_meta["Letzter Datensatz"] = zebra_meta["Letzter Datensatz"].dt.strftime("%Y-%m-%d")
    st.dataframe(zebra_meta, use_container_width=True, hide_index=True)

st.subheader("Karte")
st.caption(
    "Tipp: Nutze die Buttons über der Karte, um Trockenzeit/Regenzeit ein- oder auszublenden (ohne App-Reload)."
)
st.info(
    "📅 **Hinweis zur Saison-Definition:** "
    "Trockenzeit = Mai-Oktober (innerhalb eines Jahres). "
    "Regenzeit = November-Dezember (Vorjahr) + Januar-April (aktuelles Jahr). "
    "Durch das Saison-Zeitfenster (z.B. 2007–2008) ist die Verteilung automatisch korrekt."
)

fig = px.scatter_mapbox(
    df_f.dropna(subset=["location_lat", "location_long"]),
    lat="location_lat",
    lon="location_long",
    color="season",
    color_discrete_map={"Trockenzeit": "red", "Regenzeit": "blue"},
    category_orders={"season": ["Trockenzeit", "Regenzeit"]},
    zoom=6,
    height=650,
    hover_data=["timestamp", "individual_local_identifier", "PRECTOTCORR", "T2M"],
)

# Viewport beim Rerun behalten:
# - uirevision wechselt nur, wenn Zeitraum oder Jahreszeit wechselt
uirevision_key = f"{period_name}-{season_option}"
center_lat = float(df_f["location_lat"].dropna().mean()) if df_f["location_lat"].notna().any() else 0.0
center_lon = float(df_f["location_long"].dropna().mean()) if df_f["location_long"].notna().any() else 0.0

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=6),
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        title=dict(text="Saison", font=dict(color="black")),
        font=dict(color="black", size=12),
        orientation="h",
        x=0.01,
        xanchor="left",
        y=0.01,
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.85)",
    ),
    uirevision=uirevision_key,
)

# Client-side Toggle Buttons (kein Streamlit-Rerun)
# Wir mappen Trace-Name -> Index, damit es robust bleibt.
trace_idx = {getattr(t, "name", str(i)): i for i, t in enumerate(fig.data)}

toggle_style = dict(
    type="buttons",
    direction="right",
    y=0.99,
    yanchor="top",
    xanchor="left",
    pad=dict(r=6, t=6),
    showactive=True,
    borderwidth=1,
    font=dict(color="black"),
)

updatemenus = []
if "Trockenzeit" in trace_idx:
    updatemenus.append(
        {
            **toggle_style,
            "x": 0.01,
            "bgcolor": "rgba(255,235,235,0.95)",
            "bordercolor": "rgba(200,0,0,0.55)",
            "buttons": [
                dict(
                    label="Trockenzeit",
                    method="restyle",
                    args=[{"visible": [True]}, [trace_idx["Trockenzeit"]]],
                    args2=[{"visible": [False]}, [trace_idx["Trockenzeit"]]],
                )
            ],
        }
    )
if "Regenzeit" in trace_idx:
    updatemenus.append(
        {
            **toggle_style,
            "x": 0.12,
            "bgcolor": "rgba(235,243,255,0.95)",
            "bordercolor": "rgba(0,70,200,0.55)",
            "buttons": [
                dict(
                    label="Regenzeit",
                    method="restyle",
                    args=[{"visible": [True]}, [trace_idx["Regenzeit"]]],
                    args2=[{"visible": [False]}, [trace_idx["Regenzeit"]]],
                )
            ],
        }
    )

fig.update_layout(updatemenus=updatemenus)
fig.update_traces(marker=dict(size=4, opacity=0.75))
st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True, "displayModeBar": True},
)
