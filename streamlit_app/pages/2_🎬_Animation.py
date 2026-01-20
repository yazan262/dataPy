from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zebra Migration Animation", layout="wide")
st.title("🎬 Zebra Migration Animation")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "zebra_weather_cleaned.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data(DATA_PATH)

# Sidebar Filter
st.sidebar.header("Animation Einstellungen")
years = sorted(df["timestamp"].dt.year.dropna().unique())

# State
if "anim_year" not in st.session_state:
    st.session_state.anim_year = years[0] if years else None
if "anim_zebra_ids" not in st.session_state:
    st.session_state.anim_zebra_ids = []

def _reset_anim_zebras():
    st.session_state.anim_zebra_ids = []
    st.session_state.pop("anim_zebra_select", None)

year = st.sidebar.selectbox(
    "Jahr",
    years,
    index=(years.index(st.session_state.anim_year) if st.session_state.anim_year in years else 0),
    key="anim_year_select",
    on_change=_reset_anim_zebras,
)
st.session_state.anim_year = year

df_year = df[df["timestamp"].dt.year == year]
zebra_ids_year = sorted(df_year["individual_local_identifier"].dropna().unique())

# Standardmäßig 3 Zebras auswählen (oder weniger falls nicht verfügbar)
default_zebras = [z for z in st.session_state.anim_zebra_ids if z in zebra_ids_year]

if "anim_zebra_select" not in st.session_state:
    st.session_state.anim_zebra_select = default_zebras

zebra_ids_selected = st.sidebar.multiselect(
    "Zebra-IDs (aus gewähltem Jahr)",
    zebra_ids_year,
    default=st.session_state.anim_zebra_select,
    key="anim_zebra_select",
)
st.session_state.anim_zebra_ids = zebra_ids_selected

# Animation-Geschwindigkeit
if "anim_speed" not in st.session_state:
    st.session_state.anim_speed = 800

anim_speed = st.sidebar.slider(
    "Animation-Geschwindigkeit (ms pro Tag)",
    min_value=200,
    max_value=2000,
    value=st.session_state.anim_speed,
    step=100,
    help="Niedrigere Werte = schnellere Animation"
)
st.session_state.anim_speed = anim_speed

anim_year = st.session_state.anim_year
anim_zebras = st.session_state.anim_zebra_ids

if not anim_year:
    st.error("Keine Jahre im Datensatz gefunden.")
    st.stop()
if not anim_zebras:
    st.warning("Bitte wähle mindestens eine Zebra-ID aus (max. 3) und klicke Apply.")
    st.stop()

# Filtere Daten
df_anim = df[df["individual_local_identifier"].isin(anim_zebras) & (df["timestamp"].dt.year == anim_year)].copy()
df_anim = df_anim.sort_values("timestamp")
df_anim = df_anim.dropna(subset=["location_lat", "location_long"])

if len(df_anim) == 0:
    st.error("Keine Daten für die ausgewählten Zebras und Jahr gefunden.")
    st.stop()

# Tracking-Informationen pro Zebra
st.subheader("📊 Tracking-Daten pro Zebra")
zebra_info_cols = st.columns(len(anim_zebras))

for idx, zebra_id in enumerate(anim_zebras):
    df_zebra = df_anim[df_anim["individual_local_identifier"] == zebra_id].copy()
    
    if len(df_zebra) > 0:
        # Erstelle Datum-Spalte für dieses Zebra
        df_zebra["date"] = df_zebra["timestamp"].dt.date
        unique_days = df_zebra["date"].nunique()
        start_date = df_zebra["timestamp"].min()
        end_date = df_zebra["timestamp"].max()
        
        with zebra_info_cols[idx]:
            st.markdown(f"**🦓 {zebra_id}**")
            st.metric("Tage", f"{unique_days:,}")
            st.caption(f"📅 {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
            st.caption(f"📍 {len(df_zebra):,} GPS-Punkte")
    else:
        with zebra_info_cols[idx]:
            st.markdown(f"**🦓 {zebra_id}**")
            st.warning("Keine Daten")

st.divider()

# Erstelle Datum-Spalte für Gruppierung nach Tag
df_anim["date"] = df_anim["timestamp"].dt.date
df_anim["date"] = pd.to_datetime(df_anim["date"])

# Für jeden Tag die durchschnittliche Position pro Zebra berechnen
df_daily = df_anim.groupby(["date", "individual_local_identifier"]).agg({
    "location_lat": "mean",
    "location_long": "mean",
    "timestamp": "first"
}).reset_index()

# Sortiere nach Datum
df_daily = df_daily.sort_values("date")

# Nur Tage mit echten Messungen anzeigen (kein Auffüllen)
df_daily_full = df_daily.copy()

# Eindeutige Tage
unique_dates = sorted(df_daily_full["date"].unique())

if len(unique_dates) == 0:
    st.error("Keine Tagesdaten gefunden.")
    st.stop()

# Farben für die Zebras (zyklisch, damit beliebig viele Zebras möglich sind)
color_palette = px.colors.qualitative.Plotly
zebra_colors = {zebra_id: color_palette[i % len(color_palette)] for i, zebra_id in enumerate(anim_zebras)}

# Zentrum der Karte
center_lat = float(df_anim["location_lat"].mean())
center_lon = float(df_anim["location_long"].mean())

# Plotly Graph Objects Animation (pro Tag nur Zebras mit Daten anzeigen)
df_lookup = df_daily_full.set_index(["date", "individual_local_identifier"])

def _get_point(date_val, zebra_id):
    try:
        row = df_lookup.loc[(date_val, zebra_id)]
        return float(row["location_lat"]), float(row["location_long"])
    except KeyError:
        return None, None

# Temperatur pro Tag (für parallele Animation)
temp_by_date = df_anim.groupby(df_anim["timestamp"].dt.date)["T2M"].mean()
temp_by_date.index = pd.to_datetime(temp_by_date.index)
temp_by_date = temp_by_date.sort_index()
temp_min = float(temp_by_date.min()) if not temp_by_date.empty else 0.0
temp_max = float(temp_by_date.max()) if not temp_by_date.empty else 1.0

# Initiale Daten (erster Tag)
initial_date = unique_dates[0]
initial_data = []
for zebra_id in anim_zebras:
    lat_val, lon_val = _get_point(initial_date, zebra_id)
    initial_data.append(
        go.Scattermapbox(
            lat=[lat_val],
            lon=[lon_val],
            mode="markers",
            marker=dict(
                size=12,
                color=zebra_colors[zebra_id],
                symbol="circle",
                opacity=0.85,
            ),
            name=zebra_id,
            hovertemplate=f"<b>{zebra_id}</b><br>" +
                          "Lat: %{lat:.4f}<br>" +
                          "Lon: %{lon:.4f}<br>" +
                          f"Datum: {initial_date.strftime('%Y-%m-%d')}<extra></extra>",
            showlegend=True,
        )
    )

temp_value = float(temp_by_date.get(initial_date, temp_min)) if not temp_by_date.empty else temp_min
initial_data.append(
    go.Bar(
        x=["Temperatur"],
        y=[temp_value],
        marker_color="orange",
        text=[f"{temp_value:.2f}°C" if not temp_by_date.empty else "Keine Daten"],
        textposition="outside",
        hovertemplate=f"Datum: {initial_date.strftime('%Y-%m-%d')}<br>Temp: {temp_value:.2f}°C<extra></extra>",
        showlegend=False,
    )
)

# Frames
frames = []
trace_ids = list(range(len(anim_zebras) + 1))
for date_val in unique_dates:
    frame_data = []
    for zebra_id in anim_zebras:
        lat_val, lon_val = _get_point(date_val, zebra_id)
        frame_data.append(
            go.Scattermapbox(
                lat=[lat_val],
                lon=[lon_val],
                mode="markers",
                marker=dict(
                    size=12,
                    color=zebra_colors[zebra_id],
                    symbol="circle",
                    opacity=0.85,
                ),
                name=zebra_id,
                hovertemplate=f"<b>{zebra_id}</b><br>" +
                              "Lat: %{lat:.4f}<br>" +
                              "Lon: %{lon:.4f}<br>" +
                              f"Datum: {date_val.strftime('%Y-%m-%d')}<extra></extra>",
                showlegend=True,
            )
        )
    temp_val = float(temp_by_date.get(date_val, temp_min)) if not temp_by_date.empty else temp_min
    frame_data.append(
        go.Bar(
            x=["Temperatur"],
            y=[temp_val],
            marker_color="orange",
            text=[f"{temp_val:.2f}°C" if not temp_by_date.empty else "Keine Daten"],
            textposition="outside",
            hovertemplate=f"Datum: {date_val.strftime('%Y-%m-%d')}<br>Temp: {temp_val:.2f}°C<extra></extra>",
            showlegend=False,
        )
    )
    frames.append(go.Frame(data=frame_data, name=str(date_val), traces=trace_ids))

fig = make_subplots(
    rows=2,
    cols=1,
    specs=[[{"type": "mapbox"}], [{"type": "xy"}]],
    row_heights=[0.8, 0.2],
    vertical_spacing=0.02,
)
for i, trace in enumerate(initial_data[:-1]):
    fig.add_trace(trace, row=1, col=1)
fig.add_trace(initial_data[-1], row=2, col=1)
fig.frames = frames

# Layout + Animation Controls
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=6,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=800,
    legend=dict(
        title=dict(text="Zebras", font=dict(color="black", size=14)),
        font=dict(color="black", size=12),
        orientation="h",
        x=0.01,
        xanchor="left",
        y=0.95,
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
    ),
    yaxis=dict(range=[temp_min, temp_max], title="Temperatur (°C)"),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": anim_speed, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": int(anim_speed * 0.5), "easing": "linear"},
                        },
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                ),
            ],
            x=0.01,
            xanchor="left",
            y=0.01,
            yanchor="bottom",
            pad=dict(t=0, r=10, b=10, l=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )
    ],
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": "Datum: ", "visible": True, "xanchor": "left"},
            pad={"t": 50, "b": 10},
            steps=[
                dict(
                    args=[
                        [str(date_val)],
                        {
                            "frame": {"duration": int(anim_speed * 0.4), "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": int(anim_speed * 0.4)},
                        },
                    ],
                    label=date_val.strftime("%Y-%m-%d"),
                    method="animate",
                )
                for date_val in unique_dates
            ],
            x=0.01,
            xanchor="left",
            y=0,
            yanchor="top",
            len=0.9,
        )
    ],
)

# Info-Box
st.info(f"📅 **Zeitraum:** {unique_dates[0].strftime('%Y-%m-%d')} → {unique_dates[-1].strftime('%Y-%m-%d')} ({len(unique_dates)} Tage) | 🦓 **Zebras:** {', '.join(anim_zebras)}")
st.caption("💡 **Tipp:** Nutze die Play/Pause Buttons oder den Slider, um durch die Tage zu navigieren. Die Temperatur unten ist mit der Animation synchron.")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True, "displayModeBar": True},
)

# Zusätzliche Statistiken
st.subheader("📊 Statistiken")
col1, col2, col3 = st.columns(3)
col1.metric("Anzahl Tage", len(unique_dates))
col2.metric("Anzahl Zebras", len(anim_zebras))
col3.metric("GPS-Punkte gesamt", len(df_anim))
