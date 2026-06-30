"""
Flik Insect Analysis Dashboard
Run: python3 app.py
"""

import base64
import io
import json as _json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from PIL import Image

import db

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "output"
RESULTS_CSV = BASE_DIR / "merged_results.csv"
SENSORS_CSV = BASE_DIR / "merged_sensors.csv"
CROPS_DIR = BASE_DIR / "crops"

SENSOR_META_COLS = {"source_file", "timestamp"}
THUMB_W, THUMB_H = 80, 80
ROWS_PER_PAGE = 10
INITIAL_THUMBS = 3  # thumbnails shown per row before "Show more"

# ── Image cache ───────────────────────────────────────────────────────────────
_image_cache: dict[str, list[str]] = {}

# ── Data loading ──────────────────────────────────────────────────────────────

VERIFY_COLS = ["detection_verified", "detection_correct",
               "classification_verified", "classification_correct", "corrected_name"]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df["crop_id"] = df["track_id"].str[:8]
    for col in VERIFY_COLS:
        df[col] = pd.NA
    _apply_verifications(df)
    for col in VERIFY_COLS:
        df[col] = df[col].astype(object)
    return df


def _apply_verifications(df: pd.DataFrame):
    """Merge DB verification data into the dataframe."""
    vdata = db.load_verifications()
    for tid, vals in vdata.items():
        idx = df.index[df["track_id"] == tid]
        if idx.empty:
            continue
        for col in VERIFY_COLS:
            v = vals.get(col)
            if v is not None:
                df.loc[idx, col] = v


def load_sensor_data() -> pd.DataFrame:
    df = pd.read_csv(SENSORS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df


# ── Global state ──────────────────────────────────────────────────────────────
results_df = load_results()
sensors_df = load_sensor_data()
sensor_vars = [c for c in sensors_df.columns if c not in SENSOR_META_COLS]


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_insects(df: pd.DataFrame, insect_filter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid_counts, false_positive_counts) aggregated by hour."""
    if insect_filter != "All":
        df = df[(df["final_family"] == insect_filter) | (df["final_species"] == insect_filter)]
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")

    valid_mask = df["detection_correct"].isna() | (df["detection_correct"] == True)
    valid = df[valid_mask].groupby("hour").size().reset_index(name="count")

    fp_mask = df["detection_correct"] == False
    false_pos = df[fp_mask].groupby("hour").size().reset_index(name="count")

    return valid, false_pos


def aggregate_environment(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    sub = df[["timestamp", variable]].dropna()
    return sub.set_index("timestamp").resample("h")[variable].mean().reset_index()


# ── Image helpers (lazy — only encode when requested) ─────────────────────────

def _encode_image(path: Path) -> str:
    with Image.open(path) as img:
        img.thumbnail((THUMB_W, THUMB_H))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _get_all_crop_images(crop_id: str) -> list[str]:
    if crop_id in _image_cache:
        return _image_cache[crop_id]
    folder = CROPS_DIR / crop_id
    if not folder.is_dir():
        _image_cache[crop_id] = []
        return []
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    encoded = [_encode_image(p) for p in paths]
    _image_cache[crop_id] = encoded
    return encoded


def get_crop_images(crop_id: str, limit: int | None = INITIAL_THUMBS) -> tuple[list[str], int]:
    """Return (thumbnails up to limit, total count)."""
    all_imgs = _get_all_crop_images(crop_id)
    total = len(all_imgs)
    if limit is not None:
        return all_imgs[:limit], total
    return all_imgs, total


# ── Graph builder ─────────────────────────────────────────────────────────────

def create_graph(insect_counts, fp_counts, env_data, env_var, start_dt, end_dt):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=insect_counts["hour"], y=insect_counts["count"],
        name="Insect observations", marker_color="#4CAF50",
        yaxis="y1", opacity=0.8,
    ))
    if not fp_counts.empty:
        fig.add_trace(go.Bar(
            x=fp_counts["hour"], y=fp_counts["count"],
            name="False positives", marker_color="#E53935",
            yaxis="y1", opacity=0.8,
        ))
    if env_var and not env_data.empty:
        fig.add_trace(go.Scatter(
            x=env_data["timestamp"], y=env_data[env_var],
            name=env_var.replace("_", " ").title(),
            mode="lines", line=dict(color="#2196F3", width=2), yaxis="y2",
        ))
    fig.update_layout(
        xaxis=dict(title="Time", range=[start_dt, end_dt]),
        yaxis=dict(title="Insect Observations (per hour)", side="left"),
        yaxis2=dict(
            title=env_var.replace("_", " ").title() if env_var else "",
            side="right", overlaying="y", showgrid=False,
        ),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=60, r=60, t=40, b=60),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        barmode="group", hovermode="x unified",
    )
    return fig


# ── Review table (builds only one page of rows) ──────────────────────────────

def create_review_table(df: pd.DataFrame, page: int = 1) -> tuple[list, html.Div]:
    """Return (card rows for current page, pagination controls)."""
    total = len(df)
    total_pages = max(1, (total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    page_df = df.iloc[start_idx:end_idx]

    if page_df.empty:
        cards = [html.P("No observations match the current filters.", className="text-muted p-3")]
    else:
        cards = [_build_row_card(row) for _, row in page_df.iterrows()]

    pagination = dbc.Row(
        dbc.Col([
            html.Div([
                dbc.Button("« Prev", id="btn-prev-page", size="sm",
                           color="secondary", className="me-2",
                           disabled=(page <= 1)),
                html.Span(f"Page {page} of {total_pages}  ({total} observations)",
                          className="align-middle me-2"),
                dbc.Button("Next »", id="btn-next-page", size="sm",
                           color="secondary",
                           disabled=(page >= total_pages)),
            ], className="d-flex align-items-center justify-content-center my-2"),
        ]),
    )
    return cards, pagination


def _build_row_card(row) -> dbc.Card:
    tid = row["track_id"]
    crop_id = row["crop_id"]

    # Detection state
    det_verified = pd.notna(row.get("detection_verified")) and row["detection_verified"] == True
    det_val = row.get("detection_correct")
    det_yes = pd.notna(det_val) and det_val == True
    det_no = pd.notna(det_val) and det_val == False

    # Classification state
    cls_verified = pd.notna(row.get("classification_verified")) and row["classification_verified"] == True
    cls_val = row.get("classification_correct")
    cls_yes = pd.notna(cls_val) and cls_val == True
    cls_no = pd.notna(cls_val) and cls_val == False
    corrected = row.get("corrected_name", "")
    if pd.isna(corrected):
        corrected = ""

    thumbs, total_count = get_crop_images(crop_id, limit=INITIAL_THUMBS)
    remaining = total_count - len(thumbs)

    thumb_els = [
        html.Img(src=src, style={
            "width": f"{THUMB_W}px", "height": f"{THUMB_H}px",
            "objectFit": "cover", "margin": "2px",
            "borderRadius": "4px", "cursor": "zoom-in",
        }, className="crop-thumb", **{"data-crop": crop_id})
        for src in thumbs
    ] if thumbs else [html.Span("No images", className="text-muted small")]

    if remaining > 0:
        thumb_els.append(
            dbc.Button(
                f"+{remaining} more",
                id={"type": "btn-show-more", "index": tid},
                color="link", size="sm", className="ms-1 p-0",
            )
        )

    return dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    html.Small("Track ID", className="text-muted"),
                    html.P(tid[:8] + "…", className="mb-1 font-monospace small", title=tid),
                    html.Small("Family", className="text-muted"),
                    html.P(row.get("final_family", "—"), className="mb-1 fw-bold"),
                    html.Small("Species", className="text-muted"),
                    html.P(row.get("final_species", "—"), className="mb-1 fst-italic"),
                ], width=2),
                dbc.Col([
                    html.Small("Family conf.", className="text-muted"),
                    html.P(f"{row.get('family_confidence', 0):.2f}", className="mb-1"),
                    html.Small("Species conf.", className="text-muted"),
                    html.P(f"{row.get('species_confidence', 0):.2f}", className="mb-1"),
                    html.Small("Time", className="text-muted"),
                    html.P(str(row["timestamp"])[:16], className="mb-1 small"),
                ], width=2),
                dbc.Col(
                    html.Div(thumb_els, style={
                        "display": "flex", "flexWrap": "wrap",
                        "alignItems": "center",
                    }),
                    width=6,
                ),
                dbc.Col([
                    html.Small("Detection", className="text-muted fw-bold"),
                    html.Div([
                        dbc.Button("Yes",
                                   id={"type": "btn-det-yes", "index": tid},
                                   color="success" if det_yes else "outline-success",
                                   size="sm", className="me-1 flex-fill"),
                        dbc.Button("No",
                                   id={"type": "btn-det-no", "index": tid},
                                   color="danger" if det_no else "outline-danger",
                                   size="sm", className="flex-fill"),
                    ], className="d-flex mb-2"),
                    html.Small("Classification", className="text-muted fw-bold"),
                    html.Div([
                        dbc.Button("Yes",
                                   id={"type": "btn-cls-yes", "index": tid},
                                   color="success" if cls_yes else "outline-success",
                                   size="sm", className="me-1 flex-fill"),
                        dbc.Button("No",
                                   id={"type": "btn-cls-no", "index": tid},
                                   color="danger" if cls_no else "outline-danger",
                                   size="sm", className="flex-fill"),
                    ], className="d-flex mb-1"),
                    dbc.Input(
                        id={"type": "input-corrected", "index": tid},
                        placeholder="Correct name…",
                        size="sm", value=corrected,
                        style={"display": "block" if cls_no else "none"},
                    ),
                    dbc.Button("Save name",
                               id={"type": "btn-save-name", "index": tid},
                               color="primary", size="sm",
                               className="mt-1 w-100",
                               style={"display": "block" if cls_no else "none"}),
                ], width=2),
            ], align="center"),
        ),
        className="mb-2",
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def _update_df(track_id: str, **kwargs):
    """Update in-memory dataframe after DB write."""
    global results_df
    idx = results_df.index[results_df["track_id"] == track_id]
    for col, val in kwargs.items():
        results_df.loc[idx, col] = val


def save_detection(track_id: str, is_correct: bool):
    db.save_detection(track_id, is_correct)
    _update_df(track_id, detection_verified=True, detection_correct=is_correct)


def save_classification(track_id: str, is_correct: bool):
    db.save_classification(track_id, is_correct)
    updates = {"classification_verified": True, "classification_correct": is_correct}
    if is_correct:
        updates["corrected_name"] = pd.NA
    _update_df(track_id, **updates)


def save_corrected_name(track_id: str, name: str):
    db.save_corrected_name(track_id, name)
    _update_df(track_id, corrected_name=name)


# ── App setup ─────────────────────────────────────────────────────────────────

families = sorted(results_df["final_family"].dropna().unique().tolist())
species = sorted(results_df["final_species"].dropna().unique().tolist())
insect_options = (
    [{"label": "All insects", "value": "All"}]
    + [{"label": f"[Family] {f}", "value": f} for f in families]
    + [{"label": f"[Species] {s}", "value": s} for s in species]
)

ts_min = results_df["timestamp"].min()
ts_max = results_df["timestamp"].max()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "Flik Insect Analysis"

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    dcc.Store(id="refresh-trigger", data=0),
    dcc.Store(id="current-page", data=1),

    # Modal for enlarged images
    dbc.Modal([
        dbc.ModalBody(id="modal-body"),
    ], id="image-modal", size="xl", is_open=False),

    dbc.Row(dbc.Col(html.H2("Flik Insect Analysis Dashboard", className="my-3 text-primary"))),

    # Section 1
    dbc.Card([
        dbc.CardHeader(html.H5("Time-Series Visualisation", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Start date"),
                    dcc.DatePickerSingle(id="filter-start-date", date=ts_min.date(),
                                         display_format="YYYY-MM-DD"),
                ], md=2),
                dbc.Col([
                    html.Label("End date"),
                    dcc.DatePickerSingle(id="filter-end-date", date=ts_max.date(),
                                         display_format="YYYY-MM-DD"),
                ], md=2),
                dbc.Col([
                    html.Label("Insect"),
                    dcc.Dropdown(id="filter-insect", options=insect_options,
                                 value="All", clearable=False),
                ], md=4),
                dbc.Col([
                    html.Label("Environmental variable"),
                    dcc.Dropdown(
                        id="filter-env-var",
                        options=[{"label": v.replace("_", " ").title(), "value": v}
                                 for v in sensor_vars],
                        value=sensor_vars[0] if sensor_vars else None,
                        clearable=False),
                ], md=4),
            ], className="mb-3 g-3"),
            dcc.Graph(id="graph-timeseries", style={"height": "450px"}),
        ]),
    ], className="mb-4"),

    # Section 2
    dbc.Card([
        dbc.CardHeader(html.H5("Classification Review", className="mb-0")),
        dbc.CardBody([
            html.Div(id="review-pagination-top"),
            html.Div(id="review-table"),
            html.Div(id="review-pagination-bottom"),
        ]),
    ], className="mb-4"),

    # Toast
    dbc.Toast(
        id="toast-saved", header="Saved", is_open=False,
        dismissable=True, duration=2500,
        style={"position": "fixed", "top": 16, "right": 16, "zIndex": 9999},
    ),
], fluid=True)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _get_filtered_df(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return results_df[(results_df["timestamp"] >= start) & (results_df["timestamp"] <= end)]


@app.callback(
    Output("graph-timeseries", "figure"),
    Input("filter-start-date", "date"),
    Input("filter-end-date", "date"),
    Input("filter-insect", "value"),
    Input("filter-env-var", "value"),
    Input("refresh-trigger", "data"),
)
def update_graph(start_date, end_date, insect_filter, env_var, _trigger):
    filtered = _get_filtered_df(start_date, end_date)
    insect_counts, fp_counts = aggregate_insects(filtered, insect_filter or "All")
    sensor_filtered = sensors_df[
        (sensors_df["timestamp"] >= pd.Timestamp(start_date)) &
        (sensors_df["timestamp"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
    ]
    env_data = aggregate_environment(sensor_filtered, env_var) if env_var else pd.DataFrame()
    return create_graph(insect_counts, fp_counts, env_data, env_var or "", start_date, end_date)


@app.callback(
    Output("review-table", "children"),
    Output("review-pagination-top", "children"),
    Output("review-pagination-bottom", "children"),
    Input("filter-start-date", "date"),
    Input("filter-end-date", "date"),
    Input("current-page", "data"),
    Input("refresh-trigger", "data"),
)
def update_table(start_date, end_date, page, _trigger):
    filtered = _get_filtered_df(start_date, end_date).reset_index(drop=True)
    cards, pagination = create_review_table(filtered, page or 1)
    return cards, pagination, pagination


# Reset to page 1 when filters change
@app.callback(
    Output("current-page", "data"),
    Input("filter-start-date", "date"),
    Input("filter-end-date", "date"),
    Input("btn-prev-page", "n_clicks"),
    Input("btn-next-page", "n_clicks"),
    State("current-page", "data"),
    prevent_initial_call=True,
)
def handle_pagination(start_date, end_date, prev_clicks, next_clicks, current_page):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    current_page = current_page or 1
    if trigger_id == "btn-next-page":
        return current_page + 1
    elif trigger_id == "btn-prev-page":
        return max(1, current_page - 1)
    # Filter changed → reset to page 1
    return 1


# Detection & classification buttons
@app.callback(
    Output("refresh-trigger", "data"),
    Output("toast-saved", "is_open"),
    Output("toast-saved", "children"),
    Input({"type": "btn-det-yes", "index": dash.ALL}, "n_clicks"),
    Input({"type": "btn-det-no", "index": dash.ALL}, "n_clicks"),
    Input({"type": "btn-cls-yes", "index": dash.ALL}, "n_clicks"),
    Input({"type": "btn-cls-no", "index": dash.ALL}, "n_clicks"),
    Input({"type": "btn-save-name", "index": dash.ALL}, "n_clicks"),
    State({"type": "input-corrected", "index": dash.ALL}, "value"),
    State("refresh-trigger", "data"),
    prevent_initial_call=True,
)
def handle_buttons(det_y, det_n, cls_y, cls_n, save_n, corrected_values, current_trigger):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered[0]
    if triggered["value"] is None:
        raise dash.exceptions.PreventUpdate
    btn_info = _json.loads(triggered["prop_id"].split(".")[0])
    track_id = btn_info["index"]
    btn_type = btn_info["type"]

    if btn_type == "btn-det-yes":
        save_detection(track_id, True)
        msg = f"{track_id[:8]}… detection: Yes"
    elif btn_type == "btn-det-no":
        save_detection(track_id, False)
        msg = f"{track_id[:8]}… detection: No"
    elif btn_type == "btn-cls-yes":
        save_classification(track_id, True)
        msg = f"{track_id[:8]}… classification: Yes"
    elif btn_type == "btn-cls-no":
        save_classification(track_id, False)
        msg = f"{track_id[:8]}… classification: No"
    elif btn_type == "btn-save-name":
        # Find the matching input value
        all_ids = ctx.states_list[0]
        name = ""
        for item in all_ids:
            if item["id"]["index"] == track_id:
                name = item.get("value", "") or ""
                break
        save_corrected_name(track_id, name)
        msg = f"{track_id[:8]}… corrected to: {name}"
    else:
        raise dash.exceptions.PreventUpdate

    return (current_trigger or 0) + 1, True, msg


# "Show more" images modal
@app.callback(
    Output("image-modal", "is_open"),
    Output("modal-body", "children"),
    Input({"type": "btn-show-more", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def show_all_images(clicks):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["value"] is None:
        raise dash.exceptions.PreventUpdate
    btn_info = _json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
    track_id = btn_info["index"]
    crop_id = track_id[:8]
    all_imgs, _ = get_crop_images(crop_id, limit=None)
    imgs = [
        html.Img(src=src, style={
            "width": "120px", "height": "120px",
            "objectFit": "cover", "margin": "4px", "borderRadius": "4px",
        })
        for src in all_imgs
    ]
    return True, html.Div(imgs, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
