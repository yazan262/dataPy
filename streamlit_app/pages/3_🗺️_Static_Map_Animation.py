from pathlib import Path

import contextily as cx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pyproj
from PIL import Image
import streamlit as st


st.set_page_config(page_title="Static Map Animation", layout="wide")
st.title("Static Map Animation")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "zebra_weather_cleaned.csv"
BUFFER_LAT_DEG = 0.05
BUFFER_LON_DEG = 0.3


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def get_basemap(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    img, extent = cx.bounds2img(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        ll=True,
        source=cx.providers.OpenStreetMap.Mapnik,
    )
    return img, extent


df = load_data(DATA_PATH)

# Sidebar filters
st.sidebar.header("Filter")
years = sorted(df["timestamp"].dt.year.dropna().unique())

if "static_year" not in st.session_state:
    st.session_state.static_year = years[0] if years else None
if "static_zebra_ids" not in st.session_state:
    st.session_state.static_zebra_ids = []


def _reset_static_zebras():
    st.session_state.static_zebra_ids = []
    st.session_state.pop("static_zebra_select", None)


year = st.sidebar.selectbox(
    "Jahr",
    years,
    index=(years.index(st.session_state.static_year) if st.session_state.static_year in years else 0),
    key="static_year_select",
    on_change=_reset_static_zebras,
)
st.session_state.static_year = year

df_year = df[df["timestamp"].dt.year == year]
zebra_ids_year = sorted(df_year["individual_local_identifier"].dropna().unique())

if "static_zebra_select" not in st.session_state:
    st.session_state.static_zebra_select = []

zebra_ids_selected = st.sidebar.multiselect(
    "Zebra-IDs (aus gewähltem Jahr)",
    zebra_ids_year,
    default=st.session_state.static_zebra_select,
    key="static_zebra_select",
)
st.session_state.static_zebra_ids = zebra_ids_selected

if "static_speed" not in st.session_state:
    st.session_state.static_speed = 800

anim_speed = st.sidebar.slider(
    "Animation-Geschwindigkeit (ms pro Tag)",
    min_value=200,
    max_value=2000,
    value=st.session_state.static_speed,
    step=100,
    help="Niedrigere Werte = schnellere Animation",
)
st.session_state.static_speed = anim_speed

anim_year = st.session_state.static_year
anim_zebras = st.session_state.static_zebra_ids

if not anim_year:
    st.error("Keine Jahre im Datensatz gefunden.")
    st.stop()
if not anim_zebras:
    st.warning("Bitte wähle mindestens eine Zebra-ID aus.")
    st.stop()

# Filter data for selected zebras
df_anim = df[
    (df["timestamp"].dt.year == anim_year)
    & (df["individual_local_identifier"].isin(anim_zebras))
].copy()
df_anim = df_anim.dropna(subset=["location_lat", "location_long"])
df_anim = df_anim.sort_values("timestamp")

if df_anim.empty:
    st.error("Keine Daten für die Auswahl vorhanden.")
    st.stop()

# Daily positions
df_anim["date"] = df_anim["timestamp"].dt.date
df_anim["date"] = pd.to_datetime(df_anim["date"])
df_daily = (
    df_anim.groupby(["date", "individual_local_identifier"])
    .agg({"location_lat": "mean", "location_long": "mean"})
    .reset_index()
    .sort_values("date")
)

unique_dates = sorted(df_daily["date"].unique())
if not unique_dates:
    st.error("Keine Tagesdaten gefunden.")
    st.stop()

# Weather metrics per day (daily data)
weather_metrics = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
metric_labels = {
    "T2M": "Temp",
    "T2M_MAX": "Temp Max",
    "T2M_MIN": "Temp Min",
    "RH2M": "Feuchte",
    "WS2M": "Wind",
    "PRECTOTCORR": "Regen",
}

weather_daily = (
    df_anim.groupby(df_anim["timestamp"].dt.date)[weather_metrics]
    .mean()
    .sort_index()
)
weather_daily.index = pd.to_datetime(weather_daily.index)

metric_min = weather_daily.min()
metric_max = weather_daily.max()

def _metric_percent(metric: str, value: float) -> float:
    min_val = float(metric_min.get(metric, 0.0))
    max_val = float(metric_max.get(metric, 1.0))
    if max_val <= min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def _color_for_percent(metric: str, pct: float) -> str:
    pct = max(0.0, min(1.0, pct))
    if metric == "RH2M":
        # Hellblau -> Blau
        r = int(173 * (1.0 - pct) + 0 * pct)
        g = int(216 * (1.0 - pct) + 82 * pct)
        b = int(230 * (1.0 - pct) + 255 * pct)
        return f"rgb({r},{g},{b})"
    # Standard: Blau (low) -> Rot (high)
    r = int(255 * pct)
    g = int(80 * (1.0 - pct) + 30 * pct)
    b = int(255 * (1.0 - pct))
    return f"rgb({r},{g},{b})"

# Map bounds with buffer (mehr Puffer in Longitude für rechteckige Ansicht)
min_lat = float(df_anim["location_lat"].min()) - BUFFER_LAT_DEG
max_lat = float(df_anim["location_lat"].max()) + BUFFER_LAT_DEG
min_lon = float(df_anim["location_long"].min()) - BUFFER_LON_DEG
max_lon = float(df_anim["location_long"].max()) + BUFFER_LON_DEG

basemap_img, extent = get_basemap(min_lon, min_lat, max_lon, max_lat)
min_x, max_x, min_y, max_y = extent
basemap_pil = Image.fromarray(basemap_img)

# Coordinate transform (lat/lon -> WebMercator)
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
df_daily[["x", "y"]] = df_daily.apply(
    lambda row: pd.Series(transformer.transform(row["location_long"], row["location_lat"])),
    axis=1,
)

# Colors per zebra (cycle)
color_palette = px.colors.qualitative.Plotly
zebra_colors = {
    zebra_id: color_palette[i % len(color_palette)]
    for i, zebra_id in enumerate(anim_zebras)
}

# Data lookup for frames
df_lookup = df_daily.set_index(["date", "individual_local_identifier"])


def _get_xy(date_val, zebra_id):
    try:
        row = df_lookup.loc[(date_val, zebra_id)]
        return float(row["x"]), float(row["y"])
    except KeyError:
        return None, None


# Initial data
initial_date = unique_dates[0]
initial_traces = []
for zebra_id in anim_zebras:
    x_val, y_val = _get_xy(initial_date, zebra_id)
    initial_traces.append(
        go.Scatter(
            x=[x_val],
            y=[y_val],
            mode="markers",
            marker=dict(size=8, color=zebra_colors[zebra_id], opacity=0.9),
            name=zebra_id,
            hovertemplate=f"<b>{zebra_id}</b><br>" +
                          "X: %{x:.0f}<br>" +
                          "Y: %{y:.0f}<br>" +
                          f"Datum: {initial_date.strftime('%Y-%m-%d')}<extra></extra>",
            showlegend=True,
        )
    )

if not weather_daily.empty:
    weather_row = weather_daily.loc[initial_date] if initial_date in weather_daily.index else None
else:
    weather_row = None

weather_vals = []
weather_pct = []
weather_text = []
for metric in weather_metrics:
    if weather_row is None or pd.isna(weather_row.get(metric, np.nan)):
        weather_vals.append(np.nan)
        weather_pct.append(0.0)
        weather_text.append("NA")
    else:
        val = float(weather_row[metric])
        weather_vals.append(val)
        pct = _metric_percent(metric, val)
        weather_pct.append(pct * 100.0)
        weather_text.append(f"{val:.2f}")

initial_traces.append(
    go.Bar(
        x=[metric_labels[m] for m in weather_metrics],
        y=weather_pct,
        marker_color=[
            _color_for_percent(m, _metric_percent(m, weather_vals[i]))
            if not np.isnan(weather_vals[i])
            else "rgb(200,200,200)"
            for i, m in enumerate(weather_metrics)
        ],
        text=weather_text,
        textposition="outside",
        hovertemplate="Datum: %{customdata[0]}<br>%{x}: %{customdata[1]}<extra></extra>",
        customdata=[[initial_date.strftime("%Y-%m-%d"), weather_text[i]] for i in range(len(weather_metrics))],
        showlegend=False,
    )
)

# Frames
frames = []
trace_ids = list(range(len(anim_zebras) + 1))
for date_val in unique_dates:
    frame_traces = []
    for zebra_id in anim_zebras:
        x_val, y_val = _get_xy(date_val, zebra_id)
        frame_traces.append(
            go.Scatter(
                x=[x_val],
                y=[y_val],
                mode="markers",
                marker=dict(size=8, color=zebra_colors[zebra_id], opacity=0.9),
                name=zebra_id,
                hovertemplate=f"<b>{zebra_id}</b><br>" +
                              "X: %{x:.0f}<br>" +
                              "Y: %{y:.0f}<br>" +
                              f"Datum: {date_val.strftime('%Y-%m-%d')}<extra></extra>",
                showlegend=True,
            )
        )
    if not weather_daily.empty and date_val in weather_daily.index:
        weather_row = weather_daily.loc[date_val]
    else:
        weather_row = None

    weather_vals = []
    weather_pct = []
    weather_text = []
    for metric in weather_metrics:
        if weather_row is None or pd.isna(weather_row.get(metric, np.nan)):
            weather_vals.append(np.nan)
            weather_pct.append(0.0)
            weather_text.append("NA")
        else:
            val = float(weather_row[metric])
            weather_vals.append(val)
            pct = _metric_percent(metric, val)
            weather_pct.append(pct * 100.0)
            weather_text.append(f"{val:.2f}")

    frame_traces.append(
        go.Bar(
            x=[metric_labels[m] for m in weather_metrics],
            y=weather_pct,
            marker_color=[
                _color_for_percent(m, _metric_percent(m, weather_vals[i]))
                if not np.isnan(weather_vals[i])
                else "rgb(200,200,200)"
                for i, m in enumerate(weather_metrics)
            ],
            text=weather_text,
            textposition="outside",
            hovertemplate="Datum: %{customdata[0]}<br>%{x}: %{customdata[1]}<extra></extra>",
            customdata=[[date_val.strftime("%Y-%m-%d"), weather_text[i]] for i in range(len(weather_metrics))],
            showlegend=False,
        )
    )
    frames.append(go.Frame(data=frame_traces, name=str(date_val), traces=trace_ids))

# Build figure
fig = make_subplots(
    rows=2,
    cols=1,
    specs=[[{"type": "xy"}], [{"type": "xy"}]],
    row_heights=[0.86, 0.14],
    vertical_spacing=0.04,
)

for trace in initial_traces[:-1]:
    fig.add_trace(trace, row=1, col=1)
fig.add_trace(initial_traces[-1], row=2, col=1)
fig.frames = frames

fig.update_layout(
    xaxis=dict(range=[min_x, max_x], visible=False, scaleanchor="y", scaleratio=1),
    yaxis=dict(range=[min_y, max_y], visible=False),
    images=[
        dict(
            source=basemap_pil,
            xref="x",
            yref="y",
            x=min_x,
            y=max_y,
            sizex=max_x - min_x,
            sizey=max_y - min_y,
            sizing="stretch",
            layer="below",
        )
    ],
    yaxis2=dict(
        range=[0, 110],
        title="Wetter (%)",
        tickvals=[0, 50, 100],
        ticktext=["0%", "50%", "100%"],
    ),
    height=720,
    margin=dict(l=0, r=0, t=20, b=10),
    legend=dict(
        title=dict(text="Zebras", font=dict(color="black", size=14)),
        font=dict(color="black", size=12),
        orientation="h",
        x=0.01,
        xanchor="left",
        y=1.02,
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.85)",
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            font=dict(color="black"),
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

st.caption(
    "Tipp: Punkte erscheinen nur an Tagen mit echten Messungen. "
    "Die Wetterbalken unten sind mit der Animation synchron."
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.caption(
    "Einheiten: Temp/Temp Max/Temp Min = °C, Feuchte = %, Wind = m/s, Regen = mm/Tag"
)
