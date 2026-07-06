"""
Contour Stem Annotator
======================
CSV format:  ID, stem_id, x0, y0, x1, y1, ..., xK-1, yK-1, c0, c1, ..., cK-1

Usage:
    pip install shiny matplotlib scipy pandas numpy
    shiny run contour_annotator.py
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelmin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import io, base64
import os
#import logging
import traceback
import json
from skimage.io import imread
#from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


from shiny import App, reactive, ui, render, Inputs, Outputs, Session
#from shinywidgets import render_plotly, output_widget

#sys.path.append(os.path.abspath(".."))  # parent of notebooks/

from shape_segmentation import spline_curvature


# ── helpers ──────────────────────────────────────────────────────────────────

# def make_sync_block():
#     syncing = reactive.Value(False)

#     @contextmanager
#     def sync():
#         print('SYNC True')
#         syncing.set(True)
#         try:
#             yield
#         finally:
#             print('SYNC False')
#             syncing.set(False)

#     return syncing, sync

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{enc}"

def fig_to_b64_with_geometry(fig: plt.Figure, dpi=110):
    buf = io.BytesIO()

    fig.set_dpi(dpi)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    ax = fig.axes[0]
    fig.canvas.draw()
    bbox = ax.get_position()  # figure coordinates (0-1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    fig_w_in, fig_h_in = fig.get_size_inches()
    width_px = int(fig_w_in * dpi)
    height_px = int(fig_h_in * dpi)
    geom = {
        "dpi": dpi,
        "fig_size_in": [fig_w_in, fig_h_in],
        "img_size_px": [width_px, height_px],
        "axes": {
            "x0": bbox.x0,
            "y0": bbox.y0,
            "w": bbox.width,
            "h": bbox.height,
            "xlim": xlim,
            "ylim": ylim,
        },
    }

    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{enc}", geom

def invert_click(geom, rx, ry):
    """
    Convert normalized image coords (rx, ry) → data coords

    Parameters:
        geom : dictionary returned by renderer
        rx   : float in [0,1]
        ry   : float in [0,1]
    """

    axes = geom["axes"]
    xlim = axes["xlim"]
    ylim = axes["ylim"]

    # flip Y (image coords are top-down)
    ry = 1.0 - ry

    # figure → axes coordinates
    ax_x = (rx - axes["x0"]) / axes["w"]
    ax_y = (ry - axes["y0"]) / axes["h"]

    # clamp for safety
    #ax_x = np.clip(ax_x, 0, 1)
    #ax_y = np.clip(ax_y, 0, 1)

    # axes → data coordinates
    xdata = xlim[0] + ax_x * (xlim[1] - xlim[0])
    ydata = ylim[0] + ax_y * (ylim[1] - ylim[0])

    return xdata, ydata

def nearest_curv_min(c: np.ndarray, k_click: float, window: int = 3) -> int:
    """Return the index of the local curvature minimum closest to k_click."""
    
    print(k_click)

    # argrelmin with order=window; fall back to global min if none found
    mins = argrelmin(c, order=window, mode="wrap")[0]
    N = len(c)
    if len(mins) == 0:
        return int(np.argmin(c))
    # Distance mod N
    dists = np.abs( np.mod(mins - k_click + N/2, N) - N/2  )
    return int(mins[dists.argmin()])


def nearest_contour_point(xy: np.ndarray, x: float, y:float, window: int = 3) -> int:
    """Return the index of closest point."""
    # argrelmin with order=window; fall back to global min if none found
    dists = np.linalg.norm(xy - np.array([x, y]), axis=1)
    return int(np.argmin(dists))


def parse_csv(path: str):
    """
    Returns (df_raw, contours NxKx2, curvatures NxK, N, K).
    Raises ValueError with a descriptive message on bad format.
    """
    df = pd.read_csv(path)

    print(f"Opened {path}:")
    print(f"df.shape={df.shape}")
    print(f"df.columns={df.columns}")

    cols = [c.strip() for c in df.columns]
    #cols[0] = 'ID'
    df.columns = cols

    if "stem_id" not in cols:
        raise ValueError("CSV must contain column 'stem_id'.")

    # Detect coordinate columns  x0,y0,x1,y1,...
    xy_cols = [c for c in cols if c.startswith("x") or c.startswith("y")]
    #c_cols  = [c for c in cols if c.startswith("c")]

    K = len([c for c in cols if c.startswith("x")])
    N = len(df)

    xs = df[[f"x{k}" for k in range(K)]].values.astype(float)  # NxK
    ys = df[[f"y{k}" for k in range(K)]].values.astype(float)  # NxK
    contours   = np.stack([xs, ys], axis=-1)                    # NxKx2
    #curvatures = df[[f"c{k}" for k in range(K)]].values.astype(float)  # NxK

    annpath = path+'.ann.csv'
    try:
        ann = pd.read_csv(annpath)

        print(f'Loaded annotation {annpath}')
    except Exception as e:
        print(f'Could not load annotation {annpath}, starting fresh')
        ann = df[['filename']].copy()    
        if "id" in df.columns:
            ann["id"] = df['id']
        else:
            ann["id"] = range(ann.shape[0])
        if "stem_id" in df.columns:
            ann['stem_id'] = df['stem_id']
        else:
            ann["stem_id"] = 0
        if "ann_src" in df.columns:
            ann["ann_src"] = df['ann_src']
        else:
            ann['ann_src'] = 'default'  

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backuppath = annpath+f'.backup.{timestamp}.csv'
    ann.to_csv(backuppath, index=False)
    print(f'Saved backup to {backuppath}')

    if "id" not in ann.columns:
        print('Missing id column, init to range(N)')
        ann["id"] = range(ann.shape[0])
    if "filename" not in ann.columns:
        raise f"filename should be in .ann.csv file {annpath}"
    if "stem_id" not in ann.columns:
        print('Missing stem_id column, init to 0')
        ann["stem_id"] = 0
        ann["ann_src"] = 'default'
    if "ann_src" not in ann.columns:
        print('Missing ann_src column, init to default')
        ann["ann_src"] = 'default'
    if "dirty_contour" not in ann.columns:
        print('Missing dirty_contour column, init to 0')
        ann["dirty_contour"] = 0
    
    print(f"contours.shape={contours.shape}")
    print(f"annotations.shape={ann.shape}")
    print(f"annotations.columns={ann.columns}")

    return df, N, K, contours, ann


# ── colour palette ────────────────────────────────────────────────────────────
BG    = "#0f1117"
PANEL = "#1a1d27"
ACC   = "#5ee7df"   # teal accent
WARN  = "#f5a623"   # amber for stem highlight
TEXT  = "#d0d4e8"

# Plotly version
# def make_contour_fig(xy: np.ndarray, stem_id: int):
#     K = xy.shape[0]

#     fig = go.Figure()

#     # Scatter points (important: enables pointIndex)
#     fig.add_trace(go.Scatter(
#         x=xy[:, 0],
#         y=xy[:, 1],
#         mode="markers",
#         marker=dict(
#             size=6,
#             color=list(range(K)),
#             colorscale="Jet",
#             opacity=0.8
#         ),
#         name="contour",
#     ))

#     # Closed contour line
#     closed = np.vstack([xy, xy[:1]])
#     fig.add_trace(go.Scatter(
#         x=closed[:, 0],
#         y=closed[:, 1],
#         mode="lines",
#         line=dict(color=ACC, width=1),
#         opacity=0.3,
#         showlegend=False
#     ))

#     # Stem point highlight
#     fig.add_trace(go.Scatter(
#         x=[xy[stem_id, 0]],
#         y=[xy[stem_id, 1]],
#         mode="markers",
#         marker=dict(size=12, color=WARN),
#         name=f"stem #{stem_id}"
#     ))

#     fig.update_layout(
#         template="plotly_dark",
#         margin=dict(l=10, r=10, t=30, b=10),
#         title="Contour",
#         paper_bgcolor=PANEL,
#         plot_bgcolor=PANEL,
#         font=dict(color=TEXT),
#         showlegend=True,
#     )

#     return fig

# def make_curv_fig(c: np.ndarray, stem_id: int, K: int):
#     ks = np.arange(K)

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=ks,
#         y=c,
#         mode="lines+markers",
#         marker=dict(
#             size=6,
#             color=ks,
#             colorscale="Jet"
#         ),
#         line=dict(color=ACC, width=2),
#         name="curvature"
#     ))

#     # Stem vertical line
#     fig.add_vline(
#         x=stem_id,
#         line=dict(color=WARN, dash="dash", width=2)
#     )

#     # Highlight point
#     fig.add_trace(go.Scatter(
#         x=[stem_id],
#         y=[c[stem_id]],
#         mode="markers",
#         marker=dict(size=12, color=WARN),
#         name=f"stem k={stem_id}"
#     ))

#     fig.update_layout(
#         template="plotly_dark",
#         margin=dict(l=10, r=10, t=30, b=30),
#         title="Curvature",
#         paper_bgcolor=PANEL,
#         plot_bgcolor=PANEL,
#         font=dict(color=TEXT),
#     )

#     return fig

# MPL version
def make_contour_fig_mpl(xy: np.ndarray, stem_id: int, img: np.ndarray = None, loc = None) -> plt.Figure:
    """xy: Kx2"""

    if (img is not None):
        fig, axes = plt.subplots(2,1, figsize=(4.5, 9), facecolor=PANEL)

        axes[1].imshow(img)
        axes[1].tick_params(colors=TEXT, labelsize=7)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor=PANEL)
    ax.set_facecolor(PANEL)

    cols = range(xy.shape[0])

    ax.scatter(xy[:, 0], xy[:, 1], s=14, c=cols, alpha=0.7, linewidths=0,
               cmap = 'jet')
    # draw closed contour lightly
    closed = np.vstack([xy, xy[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=ACC, lw=0.6, alpha=0.3)
    # stem point
    ax.scatter(xy[stem_id, 0], xy[stem_id, 1], s=80, c=WARN,
               zorder=5, linewidths=0, label=f"stem #{stem_id}")
    ax.yaxis.set_inverted(True)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_aspect('equal', adjustable='datalim')
    #ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WARN,
    #          edgecolor="none", loc="upper right")
    ax.tick_params(colors=TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3250")
    if (loc is None):
        ax.set_title("Contour", color=TEXT, fontsize=10, pad=6)
    else:
        ax.set_title(f"Contour #{loc}", color=TEXT, fontsize=10, pad=6)
    fig.tight_layout(pad=0.6)
    return fig


def make_curv_fig_mpl(c: np.ndarray, stem_id: int, K: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    ks = np.arange(K)
    ax.plot(ks, c, color=ACC, lw=1.4)

    cols = ks
    ax.scatter(ks, c, c=cols, lw=1.4, cmap='jet')

    ax.fill_between(ks, c, alpha=0.12, color=ACC)
    ax.axvline(stem_id, color=WARN, lw=1.5, ls="--", label=f"stem k={stem_id}")
    ax.scatter([stem_id], [c[stem_id]], s=60, c=WARN, zorder=5, linewidths=0)
    #ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WARN,
    #          edgecolor="none", loc="upper right")
    ax.tick_params(colors=TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3250")
    ax.set_xlabel("k", color=TEXT, fontsize=8)
    ax.set_title("Curvature", color=TEXT, fontsize=10, pad=6)
    fig.tight_layout(pad=0.6)
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(f"""
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500&display=swap');
            body, .shiny-html-output {{
                background: {BG};
                color: {TEXT};
                font-family: 'IBM Plex Sans', sans-serif;
                font-size: 14px;
            }}
            h5 {{ font-family: 'IBM Plex Mono', monospace; color: {ACC}; letter-spacing:.04em; }}
            .card {{
                background: {PANEL};
                border: 1px solid #2e3250;
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 12px;
            }}
            .form-control, .form-select {{
                background: #0f1117 !important;
                color: {TEXT} !important;
                border: 1px solid #2e3250 !important;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 13px;
            }}
            .btn-primary {{
                background: {ACC} !important;
                border: none !important;
                color: #0f1117 !important;
                font-weight: 600;
                font-family: 'IBM Plex Mono', monospace;
            }}
            .btn-secondary {{
                background: #2e3250 !important;
                border: none !important;
                color: {TEXT} !important;
                font-family: 'IBM Plex Mono', monospace;
            }}
            .btn-success {{
                background: #3ecf8e !important;
                border: none !important;
                color: #0f1117 !important;
                font-weight: 700;
                font-family: 'IBM Plex Mono', monospace;
            }}
            #status_msg {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 12px;
                min-height: 1.4em;
            }}
            #item_info {{
                color: white;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 12px;
                min-height: 1.4em;
            }}
            .plot-wrap img {{ width: 100%; border-radius: 6px; }}
            input[type=range] {{ accent-color: {ACC}; }}
        """)
    ),

    ui.h5("◈  Contour Stem Annotator"),
    ui.hr(style=f"border-color:#2e3250; margin:6px 0 14px 0;"),

    # Plotly test
    #output_widget("test_plot"),

    # ── file row ──────────────────────────────────────────────────────────────
    ui.div({"class": "card"},
        ui.row(
            ui.column(9,
                ui.input_text("csv_path", None,
                              placeholder="/path/to/contours.csv",
                              value="../data/contours_clean_200_with_metadata.csv",
                              width="100%",
                              update_on="blur"),
                ui.input_text("image_dir", None,
                              placeholder="/path/to/image_dir",
                              value='/mnt/data/users/jsoto/images/',
                              width="100%",
                              update_on="blur"),
            ),
            ui.column(3,
                ui.input_action_button("open_btn", "Open", class_="btn-secondary w-50 h-50"),
                ui.input_action_button("save_btn", "Save", class_="btn-primary w-100")
            ),
        ),
        ui.output_text("status_msg"),
    ),

    # ── ID selector ──────────────────────────────────────────────────────────
    ui.div({"class": "card"},
        ui.row(
            ui.column(2, ui.input_text("id_text", "ID", value="0", width="100%")),
            ui.column(7, ui.input_slider("id_slider", None, min=0, max=1, value=0, step=1, width="100%")),
            ui.column(1, ui.input_action_button("prev_btn", "◂", class_="btn-secondary w-100")),
            ui.column(1, ui.input_action_button("next_btn", "▸", class_="btn-secondary w-100")),
            ui.column(1, ui.input_action_button("ok_btn", "V>", class_="btn-secondary w-100")),
        ),
    ),

    # ── plots ─────────────────────────────────────────────────────────────────
    ui.row(
        ui.column(6,
            ui.input_checkbox("show_image", "Show image", value=False),
            ui.div({"class": "plot-wrap"},
                ui.output_ui("contour_plot_mpl"),
                #output_widget("contour_plot"),
            ),
            # click inputs for contour: we'll use a hidden number input updated by JS
            ui.tags.input(id="contour_click_k", type="hidden", value="-1"),
        ),
        ui.column(6,
            ui.div({"class": "plot-wrap"},
                ui.output_ui("curv_plot_mpl"),
                #output_widget("curv_plot"),
            ),
            ui.tags.input(id="curv_click_k", type="hidden", value="-1"),
            ui.input_checkbox("dirty_contour", "Dirty contour", value=False),
            ui.output_text_verbatim("item_info"),
        ),
    ),

    # JS: relay matplotlib image clicks → nearest point index via Shiny.setInputValue
    # MPL click handler. Not needed when using Plotly
    ui.tags.script("""
    function attachClickHandler(imgId, inputId, K) {
        const img = document.getElementById(imgId);
        if (!img) return;
        img.style.cursor = 'crosshair';
                   
        function invert(rx, ry, geom) {
            const ax = geom.axes;

            // flip y (image coords → math coords)
            ry = 1.0 - ry;

            // figure → axes
            let ax_x = (rx - ax.x0) / ax.w;
            let ax_y = (ry - ax.y0) / ax.h;

            //ax_x = Math.min(1, Math.max(0, ax_x));
            //ax_y = Math.min(1, Math.max(0, ax_y));

            // axes → data
            const x = ax.xlim[0] + ax_x * (ax.xlim[1] - ax.xlim[0]);
            const y = ax.ylim[0] + ax_y * (ax.ylim[1] - ax.ylim[0]);

            return [x, y];
        }

        console.log("Attching onclick to",img)
        img.onclick = function(e) {
            const rect = img.getBoundingClientRect();
            const rx = (e.clientX - rect.left) / rect.width;
            const ry = (e.clientY - rect.top) / rect.height;

            console.log("CLICK","rx",rx,"ry",ry)
                   
            const geom = JSON.parse(img.dataset.geom);

            console.log("  GEOM",geom)
                   
            if (!geom) {
                   Shiny.setInputValue(inputId, {rx: rx, ry:ry}, {priority: 'event'});
                   return
            }

            const [x, y] = invert(rx, ry, geom);
                   
            console.log("  DATA COORDS","x",x,"y",y)
           
            Shiny.setInputValue(inputId, {rx: rx, ry:ry, x:x, y:y}, {priority: 'event'});
            return

            // map rx in [0,1] to k in [0, K-1]
            // matplotlib tight layout: approx 8% left margin, 2% right margin
            const left_frac = 0.11, right_frac = 0.97;
            const k_frac = (rx - left_frac) / (right_frac - left_frac);
            const k = Math.round(k_frac * (K - 1));
            const kc = Math.max(0, Math.min(K - 1, k));
        };
    }

    // Re-attach after each render
    const obs = new MutationObserver(function() {
        const K = parseInt(document.getElementById('K_val')?.value || '0');
        if (K > 0) {
            attachClickHandler('contour_img', 'contour_click_k', K);
            attachClickHandler('curv_img',    'curv_click_k',    K);
        }
    });
    obs.observe(document.body, {childList: true, subtree: true});
    """),
)


# ── Server ───────────────────────────────────────────────────────────────────

def server(input: Inputs, output: Outputs, session: Session):

    # ── state ─────────────────────────────────────────────────────────────────
    data        = reactive.Value(None)   # dict with df, N, K, contours
    cur_id      = reactive.Value(0)      # current sample index
    #cur_stem    = reactive.Value(0)      # current stem_id
    annotations = reactive.Value(None)
    show_image  = reactive.value(False)

    #syncing, sync_block = make_sync_block()

    # Reactive state

    @reactive.calc
    def cur_loc():
        ann = annotations()
        idx  = cur_id()
        if ann is None:
            return None
        return ann.index[idx]

    @reactive.calc
    def cur_stem():
        ann = annotations()
        idx  = cur_id()
        if ann is None:
            return 0
        return ann.loc[cur_loc(),"stem_id"]


    @reactive.calc
    def curvature():
        d = data()
        idx  = cur_id()
        if d is None or "error" in d:
            return None
        contour = d['contours'][idx]
        print(f"contour.shape = {contour.shape}")
        return spline_curvature(contour, smooth=0)

    # ── open file ─────────────────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.open_btn)
    def _open():
        print(f"Trying to open CSV...")
        path = input.csv_path().strip()
        try:
            df, N, K, contours, ann = parse_csv(path)
            data.set(dict(df=df, N=N, K=K, contours=contours))
            annotations.set(ann)
            cur_id.set(0)
            #cur_stem.set(int(annotations.iloc[0]["stem_id"]))
            ui.update_slider("id_slider", min=0, max=N - 1, value=0)
            ui.update_text("id_text", value="0")
            # inject K into a hidden span for JS
            ui.insert_ui(
                ui.tags.input(id="K_val", type="hidden", value=str(K)),
                selector="body", where="beforeEnd", immediate=True
            )
        except Exception as e:
            data.set({"error": str(e)})
            #logging.exception("Error opening CSV")
            traceback.print_exception(type(e), e, e.__traceback__)

    @reactive.Effect
    @reactive.event(input.save_btn)
    def _save():
        print(f"Trying to save Annotation CSV...")
        path = input.csv_path().strip()
        try:
            df = data()['df'].copy()
            ann = annotations()

            #df.loc[ann.index, ann.columns] = ann

            annpath = path+'.ann.csv'
            print(f'Saving to {annpath}...')
            ann.to_csv( annpath, index=False )
        except Exception as e:
            data.set({"error": str(e)})
            #logging.exception("Error opening CSV")
            traceback.print_exception(type(e), e, e.__traceback__)

    @output
    @render.text
    def status_msg():
        d = data()
        if d is None:
            return "No file loaded."
        if "error" in d:
            return f"⚠  {d['error']}"
        N, K = d["N"], d["K"]
        ann = annotations()
        manual_n = (ann["ann_src"]=='manual').sum()
        default_n = (ann["ann_src"]=='default').sum()
        return f"Loaded  N={N}  K={K}  |  default {default_n}/{N}, manual {manual_n}/{N}"

    @output
    @render.text
    def item_info():
        d = data()
        if d is None:
            return "No file loaded."
        if "error" in d:
            return "-"
        
        df = d['df']
        ann = annotations()
        loc = cur_loc()

        item = df.loc[loc,['filename','individual','side','area','type','color']]
        
        return f"METADATA\n{item.to_string()}\n\nANNOTATION\n{ann.loc[loc].to_string()}"

    # ── ID synchronisation: text ↔ slider ↔ prev/next → cur_id ───────────────
    # All three sources write to cur_id; cur_id drives the slider + text display.

    def set_stem_id(stem_id):
        ann = annotations().copy()
        ann.loc[cur_loc(), "stem_id"] = stem_id
        ann.loc[cur_loc(), "ann_src"] = 'manual'
        annotations.set(ann)

    def validate_stem_id():
        ann = annotations().copy()
        if (ann.loc[cur_loc(), "ann_src"] == 'default'):
            ann.loc[cur_loc(), "ann_src"] = 'valid'
        annotations.set(ann)

    @reactive.Effect
    @reactive.event(input.id_text, ignore_none=True)
    def _from_text():
        """Update cur_id from text"""
        print("_from_text")
        # if (syncing()): 
        #     print("_from_text SYNC BLOCKED")
        #     return
        d = data()
        if d is None or "error" in d: return
        try:
            v = int(input.id_text())
            v = max(0, min(d["N"] - 1, v))
        except ValueError:
            return
        if v != input.id_slider():
            ui.update_slider("id_slider", value=v)
        #if v != cur_id():
        #    cur_id.set(v)

    @reactive.Effect
    @reactive.event(input.id_slider)
    def _from_slider():
        """Update cur_id from slider"""
        print("_from_slider")
        # if (syncing()): 
        #     print("_from_slider SYNC BLOCKED")
        #     return
        d = data()
        if d is None or "error" in d: return
        v = int(input.id_slider())
        if v != cur_id():
            cur_id.set(v)

    @reactive.Effect
    @reactive.event(input.prev_btn)
    def _prev():
        """Update cur_id to previous"""
        d = data()
        if d is None or "error" in d: return
        #with reactive.isolate():
        v = max(0, input.id_slider() - 1)
        #if v != cur_id():
        if v != input.id_slider():
            ui.update_slider("id_slider", value=v)
        #cur_id.set(v)

    @reactive.Effect
    @reactive.event(input.next_btn)
    def _next():
        """Update cur_id to next"""
        d = data()
        if d is None or "error" in d: return
        #with reactive.isolate():
        v = min(d["N"] - 1, input.id_slider() + 1)
        #if v != cur_id():
        if v != input.id_slider():
            ui.update_slider("id_slider", value=v)
        #cur_id.set(v)

    @reactive.Effect
    @reactive.event(input.ok_btn)
    def _ok():
        """Validate current annotation and update cur_id to next"""
        d = data()
        if d is None or "error" in d: return
        validate_stem_id()
        
        v = min(d["N"] - 1, input.id_slider() + 1)
        #if v != cur_id():
        if v != input.id_slider():
            ui.update_slider("id_slider", value=v)

    # Keep slider + text in sync whenever cur_id changes
    @reactive.Effect
    def _sync_widgets():
        v = cur_id()
        #with sync_block():
        #ui.update_slider("id_slider", value=v)
        ui.update_text("id_text", value=str(v))

    @reactive.Effect
    def _sync_checkbox():
        show_image.set(input.show_image())

    @reactive.Effect
    @reactive.event(input.dirty_contour)
    def _from_checkbox():
        ann = annotations()
        if (ann is None):
            return
        loc = cur_loc()

        v = int(input.dirty_contour())
        item_dirty = int(ann.loc[loc, "dirty_contour"])
        if item_dirty == v:
            return
        
        print('DIRTY CHECKBOX ->',v)

        ann = ann.copy()
        ann.loc[loc, "dirty_contour"] = v
        annotations.set(ann)

    @reactive.Effect
    def _sync_checkbox_dirty_contour():
        ann = annotations()
        if (ann is None):
            return
        loc = cur_loc()

        v = int(input.dirty_contour())
        item_dirty = int(ann.loc[loc, "dirty_contour"])
        if item_dirty == v:
            return
        
        print('DIRTY CHECKBOX <-',item_dirty)

        ui.update_checkbox("dirty_contour", value=bool(item_dirty))

    # ── click handlers → find nearest curvature minimum → update stem ─────────

    # Plotly click
    # @reactive.Effect
    # @reactive.event(input.contour_plot_click)
    # def _contour_click():
    #     print("CLICK CONTOUR")

    #     d = data()
    #     if d is None or "error" in d:
    #         return

    #     event = input.contour_plot_click()
    #     if event is None:
    #         return

    #     # Option 1: exact point index (best)
    #     k_click = event["points"][0]["pointIndex"]

    #     print(f"  k_click = {k_click}")

    #     # true nearest in 2D
    #     # x_click = event["points"][0]["x"]
    #     # y_click = event["points"][0]["y"]

    #     # dists = np.linalg.norm(xy - [x_click, y_click], axis=1)
    #     # k_click = np.argmin(dists)

    #     with reactive.isolate():
    #         c = curvature()

    #     new_stem = nearest_contour_curv_min(c, k_click)

    #     print(f"  new_stem_id = {new_stem}")
 
    #     set_stem_id(new_stem)

    # @reactive.Effect
    # @reactive.event(input.curv_plot_click)
    # def _curv_click():
    #     print("CLICK CURVATURE")

    #     d = data()
    #     if d is None or "error" in d:
    #         return

    #     event = input.curv_plot_click()
    #     if event is None:
    #         return

    #     # x coordinate is k
    #     k_click = int(round(event["points"][0]["x"]))

    #     print(f"  k_click = {k_click}")

    #     with reactive.isolate():
    #         c = curvature()

    #     new_stem = nearest_curv_min(c, k_click)

    #     print(f"  new_stem_id = {new_stem}")

    #     set_stem_id(new_stem)

    # MPL click
    @reactive.Effect
    @reactive.event(input.contour_click_k)
    def _contour_click_mpl():
        d = data()
        idx = cur_id()
        if d is None or "error" in d: return
        event = input.contour_click_k()

        print("_contour_click_mpl",event)

        with reactive.isolate():
            c = curvature()
            xy = d['contours'][idx]
        new_stem = nearest_contour_point(xy, event['x'], event['y'])
        set_stem_id(new_stem)

    @reactive.Effect
    @reactive.event(input.curv_click_k)
    def _curv_click_mpl():
        d = data()
        if d is None or "error" in d: return
        event = input.curv_click_k()

        print("_curv_click_mpl",event)

        #if k_click < 0: return
        with reactive.isolate():
            c = curvature()
        new_stem = nearest_curv_min(c, event['x'])
        set_stem_id(new_stem)


    # Calculated variables

    # ── plots (both depend only on cur_id + cur_stem) ─────────────────────────

    # @output
    # @render_plotly
    # def test_plot():
    #     import plotly.graph_objects as go
    #     fig = go.Figure(go.Scatter(
    #         x=[1,2,3],
    #         y=[1,4,9],
    #         mode="markers"
    #     ))
    #     return fig

    # @reactive.Effect
    # @reactive.event(input.test_plot_click)
    # def _test_click():
    #     print('TEST CLICK')
    #     print(input.test_plot_click())


    # Plotly plots
    # @output
    # @render_plotly
    # def contour_plot():
    #     d = data()
    #     if d is None or "error" in d:
    #         return go.Figure()

    #     idx  = cur_id()
    #     stem = cur_stem()
    #     xy   = d["contours"][idx]

    #     return make_contour_fig(xy, stem)

    # @output
    # @render_plotly
    # def curv_plot():
    #     d = data()
    #     if d is None or "error" in d:
    #         return go.Figure()

    #     idx  = cur_id()
    #     stem = cur_stem()
    #     K    = d["K"]
    #     c    = curvature()

    #     return make_curv_fig(c, stem, K)

    # MPL plots
    @output
    @render.ui
    def contour_plot_mpl():
        d = data()
        if d is None or "error" in d:
            return ui.p("No data.", style="color:#888; padding:2em;")
        idx  = cur_id()
        loc = cur_loc()
        stem = cur_stem()
        simg = show_image()

        img = None
        if (simg):
            img_dir = Path(input.image_dir().strip())

            imgname = d['df'].iloc[idx].filename
            img_fullpath = str(img_dir / imgname)
            print('IMG PATH',img_fullpath)
            img = imread(img_fullpath) 

        xy   = d["contours"][idx]           # Kx2
        fig  = make_contour_fig_mpl(xy, stem, img, loc)
        src, geom  = fig_to_b64_with_geometry(fig)
        return ui.tags.img(id="contour_img", src=src,
                           style="width:100%;border-radius:6px;cursor:crosshair;",
                           **{"data-geom": json.dumps(geom)})

    @output
    @render.ui
    def curv_plot_mpl():
        d = data()
        if d is None or "error" in d:
            return ui.p("No data.", style="color:#888; padding:2em;")
        idx  = cur_id()
        stem = cur_stem()
        K    = d['K']
        c    = curvature()
        fig  = make_curv_fig_mpl(c, stem, K)
        src, geom  = fig_to_b64_with_geometry(fig)
        return ui.tags.img(id="curv_img", src=src,
                           style="width:100%;border-radius:6px;cursor:crosshair;",
                           **{"data-geom": json.dumps(geom)})


app = App(app_ui, server)