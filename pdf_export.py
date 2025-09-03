from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import datetime
import json
import os
import tempfile
import traceback

import pandas as pd
import polars as pl


# Import sdílené logiky – pouze z core_logic (žádná závislost na player_rating)
import core_logic as cl


BASE_DIR: Path = Path(__file__).parent.resolve()
AVG_DATA_DIR: Path = BASE_DIR / "AVG_Parquet"
COL_POS: str = "Converted Position"
MIN_MINUTES: int = 300
TOP_CLUBS: List[str] = ["Slavia Praha", "Sparta Praha", "Viktoria Plzeň"]


def _load_combined_avg_dataframe() -> pd.DataFrame:
    """Načte a připraví kombinovaný dataframe průměrů ze všech Parquet souborů.

    Využívá konstanty a funkce z importovaného modulu `pr` (player_rating).
    """
    all_avg_data_frames: list[pd.DataFrame] = []

    avg_dir: Path = AVG_DATA_DIR
    for parquet_file in avg_dir.glob("*.parquet"):
        all_avg_data_frames.append(pd.read_parquet(parquet_file, engine="pyarrow"))

    if not all_avg_data_frames:
        return pd.DataFrame()

    df_combined_avg = pd.concat(all_avg_data_frames, ignore_index=True)

    # Odstranění GK, výpočty pomocných metrik, převod číselných sloupců
    if "Converted Position" in df_combined_avg.columns:
        df_combined_avg = df_combined_avg[df_combined_avg["Converted Position"] != "GK"]

    df_combined_avg = cl.compute_loose_ball_duels(df_combined_avg)
    df_combined_avg = cl.compute_effective_passing_index(df_combined_avg)

    for col in df_combined_avg.columns:
        if col in ["Player", "Team", COL_POS]:
            continue
        if df_combined_avg[col].dtype == "object":
            try:
                df_combined_avg[col] = pd.to_numeric(
                    df_combined_avg[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace("%", "", regex=False),
                    errors="coerce",
                )
            except Exception:
                # ponecháme, pokud převod nejde bezpečně
                pass
        else:
            df_combined_avg[col] = pd.to_numeric(df_combined_avg[col], errors="coerce")

    return df_combined_avg


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Position' in df.columns:
        df['Position'] = df['Position'].astype(str).apply(lambda x: x.split(',')[0].strip())

    def convert_position(pos: str) -> str:
        mappings = {
            'ST': 'FC', 'CF': 'FC', 'FW': 'FC', 'STRIKER': 'FC', 'FORWARD': 'FC', 'CENTRE FORWARD': 'FC', 'CENTER FORWARD': 'FC',
            'RB': 'DR', 'RIGHT BACK': 'DR', 'RWB': 'DR', 'RIGHT WING BACK': 'DR', 'RWBK': 'DR',
            'LB': 'DL', 'LEFT BACK': 'DL', 'LWB': 'DL', 'LEFT WING BACK': 'DL', 'LWBK': 'DL',
            'RW': 'ARML', 'RWF': 'ARML', 'RAMF': 'ARML', 'RIGHT WINGER': 'ARML', 'RIGHT WING': 'ARML', 'RWG': 'ARML',
            'LW': 'ARML', 'LWF': 'ARML', 'LAMF': 'ARML', 'LEFT WINGER': 'ARML', 'LEFT WING': 'ARML', 'LWG': 'ARML',
            'RM': 'ARML', 'RMF': 'ARML', 'RIGHT MIDFIELDER': 'ARML',
            'LM': 'ARML', 'LMF': 'ARML', 'LEFT MIDFIELDER': 'ARML',
            'AMF': 'AMC', 'CAM': 'AMC', 'ATTACKING MIDFIELDER': 'AMC',
            'CB': 'DC', 'LCB': 'DC', 'RCB': 'DC', 'CENTRE BACK': 'DC', 'CENTER BACK': 'DC',
            'DMF': 'DMC', 'CDM': 'DMC', 'RDMF': 'DMC', 'LDMF': 'DMC', 'DEFENSIVE MIDFIELDER': 'DMC',
            'CMF': 'MC', 'CM': 'MC', 'RCMF': 'MC', 'LCMF': 'MC', 'CENTRE MIDFIELDER': 'MC', 'CENTER MIDFIELDER': 'MC',
            'GK': 'GK',
        }
        return mappings.get(pos.upper(), pos)

    if 'Position' in df.columns:
        df['Converted Position'] = df['Position'].apply(convert_position)
        if 'Team' in df.columns:
            if 'Converted Position' in df.columns.tolist() and df.columns.tolist().count('Converted Position') > 1:
                df = df.loc[:, ~df.columns.duplicated()]
            team_index = df.columns.get_loc("Team")
            df.insert(team_index + 1, 'Converted Position', df.pop('Converted Position'))

    df.rename(columns=lambda x: x.replace(", %", ""), inplace=True)
    df.rename(columns=lambda x: x.replace(", m", ""), inplace=True)
    return df


def _load_logic_json() -> Dict[str, Any]:
    logic_path = BASE_DIR / "metric_logic.json"
    with open(logic_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_and_process_data_for_pdf(player: str, df_player_data_full: pd.DataFrame, df_combined_avg: pd.DataFrame) -> Optional[Dict[str, Any]]:
    rows_p_initial = df_player_data_full[df_player_data_full["Player"] == player].copy()
    if rows_p_initial.empty:
        return None

    player_positions = rows_p_initial[COL_POS].dropna().astype(str).unique().tolist()
    if not player_positions:
        return None
    main_position = player_positions[0]
    positions_for_display = player_positions

    rows_p_processed = cl.compute_loose_ball_duels(rows_p_initial)
    rows_p_processed = cl.compute_effective_passing_index(rows_p_processed)
    for col in rows_p_processed.columns:
        if col not in ["Player", "Team", COL_POS] and rows_p_processed[col].dtype == 'object':
            try:
                rows_p_processed[col] = pd.to_numeric(rows_p_processed[col].astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False), errors='coerce')
            except Exception:
                pass
        elif col not in ["Player", "Team", COL_POS]:
            rows_p_processed[col] = pd.to_numeric(rows_p_processed[col], errors='coerce')

    rows_p_filtered = rows_p_processed[rows_p_processed["Minutes played"] >= MIN_MINUTES].copy()
    if rows_p_filtered.empty:
        return None

    player_row = rows_p_filtered.iloc[0]
    player_club = player_row.get("Team", "N/A")
    player_minutes = player_row.get("Minutes played", 0)
    report_date = datetime.date.today().strftime("%d.%m.%Y")
    player_age = player_row.get("Age"); player_height = player_row.get("Height"); player_weight = player_row.get("Weight")

    df_min_combined_avg = df_combined_avg[df_combined_avg["Minutes played"] >= MIN_MINUTES].copy()
    positions_for_filter = cl.get_positions_for_avg_filter(main_position)
    df_pos_filtered = df_min_combined_avg[df_min_combined_avg[COL_POS].isin(positions_for_filter)]
    if df_pos_filtered.empty:
        return None

    p_avg = rows_p_filtered.mean(numeric_only=True)

    numeric_cols = df_pos_filtered.select_dtypes(include=["number"]).columns.tolist()
    base_cols = [c for c in ["Player", "Team"] if c in df_pos_filtered.columns]
    use_cols = base_cols + numeric_cols
    if not numeric_cols:
        return None

    pl_df = pl.from_pandas(df_pos_filtered[use_cols])
    lg_avg_row = pl_df.select([pl.col(c).mean().alias(c) for c in numeric_cols]).to_pandas().iloc[0]
    lg_avg = pd.Series(lg_avg_row, index=numeric_cols, dtype="float64")
    top_pl = pl_df.filter(pl.col("Team").is_in(TOP_CLUBS)) if "Team" in pl_df.columns else pl.DataFrame()
    if top_pl.is_empty():
        tp_avg = pd.Series(pd.NA, index=lg_avg.index)
    else:
        tp_avg_row = top_pl.select([pl.col(c).mean().alias(c) for c in numeric_cols]).to_pandas().iloc[0]
        tp_avg = pd.Series(tp_avg_row, index=numeric_cols, dtype="float64")

    logic_data = _load_logic_json()
    positions_for_weights = ["AMR", "AML"] if main_position in ["AMR", "AML", "ARML"] else (["DR", "DL"] if main_position in ["DR", "DL"] else [main_position])
    logic_df = cl.logic_flat_df(positions_for_weights, logic_data)

    rat_lg = cl.rating_series(p_avg, lg_avg)
    rat_tp = cl.rating_series(p_avg, tp_avg)
    logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_lg.index and m in rat_tp.index]
    score_lg = cl.weighted_score(rat_lg[logic_metrics_in_data], logic_df)
    score_tp = cl.weighted_score(rat_tp[logic_metrics_in_data], logic_df)
    sec_lg_df, sub_lg_df = cl.breakdown_scores(rat_lg, main_position, logic_data)
    sec_tp_df, sub_tp_df = cl.breakdown_scores(rat_tp, main_position, logic_data)

    all_metrics_rows = []
    for m in sorted(list(p_avg.index)):
        player_val = p_avg.get(m, pd.NA)
        lg_val = lg_avg.get(m, pd.NA)
        tp_val = tp_avg.get(m, pd.NA)
        val_vs_lg_perc = (player_val / lg_val * 100) if pd.notna(player_val) and pd.notna(lg_val) and lg_val != 0 else pd.NA
        val_vs_tp_perc = (player_val / tp_val * 100) if pd.notna(player_val) and pd.notna(tp_val) and tp_val != 0 else pd.NA
        all_metrics_rows.append({
            "Metric": m,
            "Player": player_val,
            "League Avg": lg_val,
            "TOP Clubs Avg": tp_val,
            "vs. League (%)": val_vs_lg_perc,
            "vs. TOP Clubs (%)": val_vs_tp_perc
        })
    df_all_numeric_ratings_for_pdf = pd.DataFrame(all_metrics_rows)

    return {
        "player_name": player,
        "player_club": player_club,
        "player_positions": positions_for_display,
        "player_minutes": int(player_minutes) if pd.notna(player_minutes) else 0,
        "report_date": report_date,
        "main_position": main_position,
        "overall_score_lg": round(score_lg, 2) if pd.notna(score_lg) else score_lg,
        "overall_score_tp": round(score_tp, 2) if pd.notna(score_tp) else score_tp,
        "sec_lg_df": sec_lg_df,
        "sec_tp_df": sec_tp_df,
        "sub_lg_df": sub_lg_df,
        "sub_tp_df": sub_tp_df,
        "df_all_numeric_ratings": df_all_numeric_ratings_for_pdf,
        "player_age": player_age,
        "player_height": player_height,
        "player_weight": player_weight,
    }

# ====== Grafická a PDF vrstva (přesunuto z player_rating.py) ======
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

try:
    import streamlit as st  # pro jednotné hlášení chyb jako v player_analysis
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False


# Cesty a fonty
PROMPT_TEMPLATE_FILE = BASE_DIR / "gemini_prompt_template.txt"
PATH_FONT_BARLOW_REGULAR = str(BASE_DIR / "fonts/BarlowSemiCondensed-Regular.ttf")
PATH_FONT_BARLOW_BOLD = str(BASE_DIR / "fonts/BarlowSemiCondensed-Bold.ttf")
# Cloud Run má zapisovatelný pouze /tmp. Rozlišíme prostředí dle proměnné K_SERVICE.
IS_CLOUD_RUN = bool(os.getenv("K_SERVICE"))
TEMP_IMG_DIR = os.path.join(tempfile.gettempdir(), "rbr_gauge_images_pdf") if IS_CLOUD_RUN else str(BASE_DIR / "temp_gauge_images_pdf_v11")

# Gemini (sjednocená inicializace jako v player_analysis)
SERVICE_ACCOUNT_JSON_PDF = "/Users/krejda/Documents/Python/service-account-key.json"
PROJECT_ID_PDF, LOCATION_PDF, MODEL_NAME_PDF = "performance-445519", "us-central1", "gemini-2.5-pro"
GEMINI_TRANSLATION_SEPARATOR = "--- ENGLISH TRANSLATION ---"

# Názvy pro tajné klíče ve všech prostředích – stejné jako v player_analysis
ENV_SECRET_NAME = "GCP_SA_JSON"
STREAMLIT_SECRET_NAME = "gcp_service_account"
LOCAL_SECRET_PATH = "inside-data-story-af484f6c4b69.json"


def initialize_gemini_pdf() -> tuple[GenerativeModel | None, bool]:
    creds = None
    secret_info = None

    env_val = os.environ.get(ENV_SECRET_NAME)
    if env_val:
        try:
            secret_info = json.loads(env_val)
            print("--- INFO: Nalezen klíč v proměnné prostředí (ENV). ---")
        except json.JSONDecodeError as e:
            if _HAS_STREAMLIT:
                st.error(f"Chyba při parsování JSON z proměnné prostředí '{ENV_SECRET_NAME}': {e}")
            else:
                print(f"Chyba při parsování JSON z ENV: {e}")
            return None, False
    elif os.path.exists(LOCAL_SECRET_PATH):
        try:
            with open(LOCAL_SECRET_PATH) as f:
                secret_info = json.load(f)
            print(f"--- INFO: Nalezen klíč v lokálním souboru '{LOCAL_SECRET_PATH}'. ---")
        except Exception as e:
            if _HAS_STREAMLIT:
                st.error(f"Chyba při čtení lokálního souboru s klíčem: {e}")
            else:
                print(f"Chyba při čtení lokálního souboru s klíčem: {e}")
            return None, False
    elif _HAS_STREAMLIT and hasattr(st, 'secrets') and STREAMLIT_SECRET_NAME in st.secrets:
        secret_info = dict(st.secrets[STREAMLIT_SECRET_NAME])
        print("--- INFO: Nalezen klíč ve Streamlit Secrets. ---")

    if secret_info:
        try:
            creds = service_account.Credentials.from_service_account_info(secret_info)
        except Exception as e:
            if _HAS_STREAMLIT:
                st.error(f"Chyba při vytváření přihlašovacích údajů z nalezeného klíče: {e}")
            else:
                print(f"Chyba při vytváření přihlašovacích údajů: {e}")
            return None, False
    else:
        msg = (
            "Chybí přihlašovací údaje pro Google Cloud! Zkontrolujte nastavení pro vaše prostředí:\n\n"
            f"- Pro Google Cloud Run: nastavte secret v ENV jako '{ENV_SECRET_NAME}'.\n"
            f"- Pro lokální spuštění: ujistěte se, že existuje soubor '{LOCAL_SECRET_PATH}'.\n"
            f"- Pro Streamlit Cloud: přidejte secret s názvem '{STREAMLIT_SECRET_NAME}'."
        )
        if _HAS_STREAMLIT:
            st.error(msg)
        else:
            print(msg)
        return None, False

    try:
        vertexai.init(project=PROJECT_ID_PDF, location=LOCATION_PDF, credentials=creds)
        model = GenerativeModel(MODEL_NAME_PDF)
        print("--- INFO: Vertex AI (PDF) úspěšně inicializováno. ---")
        return model, True
    except Exception as e:
        warn = f"Klíč byl načten, ale selhala inicializace Vertex AI: {e}"
        if _HAS_STREAMLIT:
            st.warning(warn)
        else:
            print(warn)
        return None, False


_gemini_pdf_renderer, _gemini_pdf_renderer_available = initialize_gemini_pdf()


# Styly PDF
FONT_BARLOW_REGULAR_NAME = 'BarlowSemiCondensed-Regular'
FONT_BARLOW_BOLD_NAME = 'BarlowSemiCondensed-Bold'
DEFAULT_FONT_NORMAL = "Helvetica"; DEFAULT_FONT_BOLD = "Helvetica-Bold"
FONT_NORMAL = DEFAULT_FONT_NORMAL; FONT_BOLD = DEFAULT_FONT_BOLD
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if Path(PATH_FONT_BARLOW_REGULAR).exists() and Path(PATH_FONT_BARLOW_BOLD).exists():
        pdfmetrics.registerFont(TTFont(FONT_BARLOW_REGULAR_NAME, PATH_FONT_BARLOW_REGULAR))
        pdfmetrics.registerFont(TTFont(FONT_BARLOW_BOLD_NAME, PATH_FONT_BARLOW_BOLD))
        pdfmetrics.registerFontFamily(FONT_BARLOW_REGULAR_NAME, normal=FONT_BARLOW_REGULAR_NAME, bold=FONT_BARLOW_BOLD_NAME)
        FONT_NORMAL = FONT_BARLOW_REGULAR_NAME
        FONT_BOLD = FONT_BARLOW_BOLD_NAME
except Exception:
    FONT_NORMAL = DEFAULT_FONT_NORMAL
    FONT_BOLD = DEFAULT_FONT_BOLD

styles_pdf = getSampleStyleSheet()
style_h1_pdf = ParagraphStyle('Heading1', parent=styles_pdf['h1'], fontSize=20, alignment=TA_CENTER, spaceAfter=0.3*cm, fontName=FONT_BOLD)
style_h2_pdf = ParagraphStyle('Heading2', parent=styles_pdf['h2'], fontSize=16, alignment=TA_LEFT, spaceBefore=0.5*cm, spaceAfter=0.2*cm, fontName=FONT_BOLD, leading=20)
style_h3_pdf = ParagraphStyle('Heading3', parent=styles_pdf['h3'], fontSize=13, alignment=TA_LEFT, spaceBefore=0.3*cm, spaceAfter=0.15*cm, fontName=FONT_BOLD)
style_normal_pdf = ParagraphStyle('NormalPDF', parent=styles_pdf['Normal'], fontSize=10, fontName=FONT_NORMAL, leading=13)
style_table_header_pdf = ParagraphStyle('TableHeaderPDF', parent=style_normal_pdf, alignment=TA_CENTER, fontName=FONT_BOLD, fontSize=9, leading=16)
style_table_cell_left_pdf = ParagraphStyle('TableCellLeftPDF', parent=style_normal_pdf, alignment=TA_LEFT, fontSize=8, fontName=FONT_NORMAL, leading=18)
style_table_cell_center_pdf = ParagraphStyle('TableCellCenterPDF', parent=style_normal_pdf, alignment=TA_CENTER, fontSize=8, fontName=FONT_NORMAL, leading=10)
style_section_name_cell_pdf = ParagraphStyle('SectionNameCellPDF', parent=style_normal_pdf, alignment=TA_LEFT, fontSize=10, fontName=FONT_NORMAL, leading=22)
color_good_pdf = colors.Color(144/255, 238/255, 144/255, alpha=0.75)
color_avg_pdf = colors.Color(255/255, 235/255, 153/255, alpha=0.75)
color_bad_pdf = colors.Color(255/255, 153/255, 153/255, alpha=0.75)
color_header_bg_pdf = colors.Color(235/255,235/255,235/255)


def build_gemini_prompt_from_processed_data_for_pdf(player_info_for_prompt: Dict[str, Any]) -> str:
    """Vytvoří prompt přesně dle `core_logic.build_prompt`."""
    player_name: str = player_info_for_prompt.get('player_name', 'Neznámý hráč')
    positions: List[str] = player_info_for_prompt.get('player_positions', ['N/A'])

    # Sekce – připrav tabulku s požadovanými názvy sloupců
    sec_lg_df = player_info_for_prompt.get("sec_lg_df", pd.DataFrame())
    sec_tp_df = player_info_for_prompt.get("sec_tp_df", pd.DataFrame())
    if sec_lg_df is None: sec_lg_df = pd.DataFrame()
    if sec_tp_df is None: sec_tp_df = pd.DataFrame()
    if not sec_lg_df.empty:
        sec_lg_df = sec_lg_df.rename(columns={"Score": "vs. League"})
    if not sec_tp_df.empty:
        sec_tp_df = sec_tp_df.rename(columns={"Score": "vs. TOP 3"})
    if not sec_lg_df.empty or not sec_tp_df.empty:
        sec_tbl = pd.merge(sec_lg_df[["Section", "vs. League"]] if not sec_lg_df.empty else pd.DataFrame(columns=["Section", "vs. League"]),
                           sec_tp_df[["Section", "vs. TOP 3"]] if not sec_tp_df.empty else pd.DataFrame(columns=["Section", "vs. TOP 3"]),
                           on="Section", how="outer")
    else:
        sec_tbl = pd.DataFrame(columns=["Section", "vs. League", "vs. TOP 3"])

    # Sub-sekce – připrav tabulku s požadovanými názvy sloupců
    sub_lg_df = player_info_for_prompt.get("sub_lg_df", pd.DataFrame())
    sub_tp_df = player_info_for_prompt.get("sub_tp_df", pd.DataFrame())
    if sub_lg_df is None: sub_lg_df = pd.DataFrame()
    if sub_tp_df is None: sub_tp_df = pd.DataFrame()
    if not sub_lg_df.empty:
        sub_lg_df = sub_lg_df.rename(columns={"Score": "vs. League"})
    if not sub_tp_df.empty:
        sub_tp_df = sub_tp_df.rename(columns={"Score": "vs. TOP 3"})
    if not sub_lg_df.empty or not sub_tp_df.empty:
        sub_tbl = pd.merge(sub_lg_df[["Section", "Subsection", "vs. League"]] if not sub_lg_df.empty else pd.DataFrame(columns=["Section", "Subsection", "vs. League"]),
                           sub_tp_df[["Section", "Subsection", "vs. TOP 3"]] if not sub_tp_df.empty else pd.DataFrame(columns=["Section", "Subsection", "vs. TOP 3"]),
                           on=["Section", "Subsection"], how="outer")
    else:
        sub_tbl = pd.DataFrame(columns=["Section", "Subsection", "vs. League", "vs. TOP 3"])

    # Detailní metriky – mapuj názvy sloupců
    df_all = player_info_for_prompt.get('df_all_numeric_ratings', pd.DataFrame()).copy()
    if df_all is None:
        df_all = pd.DataFrame()
    if not df_all.empty:
        # Vylouč explicitně problematickou metriku
        if 'Metric' in df_all.columns:
            df_all = df_all[df_all['Metric'] != 'Aerial duels per 90.1']
        all_metrics_tbl = df_all.rename(columns={
            'Player': 'Hráč',
            'League Avg': 'Liga Ø',
            'vs. League (%)': 'vs. League',
            'vs. TOP Clubs (%)': 'vs. TOP 3',
        })
        # ponecháme jen relevantní sloupce, pokud existují
        keep_cols = [c for c in ['Metric', 'Hráč', 'Liga Ø', 'vs. League', 'vs. TOP 3'] if c in all_metrics_tbl.columns]
        all_metrics_tbl = all_metrics_tbl[keep_cols]
    else:
        all_metrics_tbl = pd.DataFrame(columns=['Metric', 'Hráč', 'Liga Ø', 'vs. League', 'vs. TOP 3'])

    # Vygeneruj prompt dle core_logic.build_prompt
    try:
        return cl.build_prompt(player_name, positions, sec_tbl, sub_tbl, all_metrics_tbl)
    except Exception as e:
        # Záložní jednoduchý prompt, kdyby se něco pokazilo
        return f"Analýza pro {player_name} ({', '.join(positions)}). Nelze sestavit tabulky pro prompt: {e}"


def get_gemini_analysis_for_pdf(player_data_for_gemini: Dict[str, Any]) -> Dict[str, str]:
    default_texts = {'cs': "Textová analýza Gemini AI není k dispozici.", 'en': "Gemini AI textual analysis is not available."}
    if not _gemini_pdf_renderer_available or not _gemini_pdf_renderer:
        return default_texts
    try:
        prompt = build_gemini_prompt_from_processed_data_for_pdf(player_data_for_gemini)
        msg = Content(role="user", parts=[Part.from_text(prompt)])
        generation_config = GenerationConfig(max_output_tokens=8192, temperature=0.6, top_p=0.9, top_k=35)
        response = _gemini_pdf_renderer.generate_content([msg], generation_config=generation_config)
        full_text = response.text
        if GEMINI_TRANSLATION_SEPARATOR in full_text:
            parts = full_text.split(GEMINI_TRANSLATION_SEPARATOR, 1)
            return {'cs': parts[0].strip(), 'en': parts[1].strip() if len(parts) > 1 else "Chyba: Anglický překlad v textu Gemini nenalezen."}
        else:
            return {'cs': full_text.strip(), 'en': "Chyba: Oddělovač pro anglický překlad v textu Gemini nenalezen."}
    except Exception as e:
        print(f"❌ Chyba při generování analýzy pomocí Gemini (pro PDF): {e}")
        traceback.print_exc()
        return default_texts


def value_to_angle_gauge(value: float, scale_min: float, scale_max: float, angle_min_map: float = 0, angle_max_map: float = 360) -> float:
    if pd.isna(value):
        return angle_min_map
    value = max(min(value, scale_max), scale_min)
    if (scale_max - scale_min) == 0:
        return angle_min_map
    return angle_min_map + (value - scale_min) / (scale_max - scale_min) * (angle_max_map - angle_min_map)


ZONE_COLORS_HEX = {"bad": "#FF9999", "avg": "#FFEE99", "good": "#99FF99", "default": "#D3D3D3"}


def get_value_color_for_donut(value: Optional[float], scale_max_val: float) -> str:
    if pd.isna(value):
        return ZONE_COLORS_HEX["default"]
    if value < 90:
        return ZONE_COLORS_HEX["bad"]
    elif value < 105:
        return ZONE_COLORS_HEX["avg"]
    else:
        return ZONE_COLORS_HEX["good"]


def create_gauge_chart_pdf(value: float, title_text: str, output_filename: str, size: str = "normal"):
    if size == "small":
        fig_size = (2.0, 2.0); outer_radius_factor = 0.42; donut_thickness_ratio = 0.35
        value_font_size = 9; title_font_size = 7; title_y_pos = 0.08
    else:
        fig_size = (2.5, 2.5); outer_radius_factor = 0.45; donut_thickness_ratio = 0.35
        value_font_size = 13; title_font_size = 8; title_y_pos = 0.10
    fig, ax = plt.subplots(figsize=fig_size); ax.set_aspect('equal'); ax.axis('off')
    center_x, center_y = 0.5, 0.5; outer_radius = outer_radius_factor; donut_thickness = outer_radius * donut_thickness_ratio
    scale_min_val, scale_max_val = 0, 150
    ax.add_patch(patches.Wedge((center_x, center_y), outer_radius, 0, 360, width=donut_thickness, facecolor='#E0E0E0', zorder=1))
    if pd.notna(value):
        value_color = get_value_color_for_donut(value, scale_max_val)
        angle_at_value = value_to_angle_gauge(value, scale_min_val, scale_max_val, 0, 360)
        ax.add_patch(patches.Wedge((center_x, center_y), outer_radius, 0, angle_at_value, width=donut_thickness, facecolor=value_color, zorder=2))
        value_display_text = f"{value:.0f}%"
    else:
        value_display_text = "N/A"
    ax.text(center_x, center_y, value_display_text, ha='center', va='center', fontsize=value_font_size, fontweight='bold', color='black', zorder=3)
    fig.text(0.5, title_y_pos, title_text, ha='center', va='bottom', fontsize=title_font_size, wrap=True, linespacing=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); plt.savefig(output_filename, bbox_inches='tight', dpi=140); plt.close(fig)


def _render_gemini_to_story(story: list[Any], text: str) -> None:
    """Přehledně naformátuje text Gemini: podnadpisy, seznamy, odstavce."""
    lines = text.split('\n')
    bullet_buf: list[tuple[str, str]] = []  # (type, content) type: 'bullet'|'enum'

    def flush_bullets():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        items = []
        for _, raw in bullet_buf:
            txt = raw.strip()
            txt = re.sub(r'\*\*(?=\S)([^\*]+?)(?<=\S)\*\*', r'<b>\1</b>', txt)
            txt = re.sub(r'_(?=\S)(.+?)(?<=\S)_', r'<i>\1</i>', txt)
            items.append(ListItem(Paragraph(txt, style_normal_pdf)))
        # Vynuceně použij klasické odrážky a nikdy nečíslované seznamy
        story.append(ListFlowable(
            items,
            bulletType='bullet',
            leftIndent=10,
            bulletFontName=FONT_NORMAL,
            bulletFontSize=style_normal_pdf.fontSize
        ))
        story.append(Spacer(1, 0.15*cm))
        bullet_buf = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_bullets(); story.append(Spacer(1, 0.10*cm)); continue
        if re.match(r'^Strana\s+\d+', line):
            continue
        if line.startswith('### '):
            header = line[4:].strip()
            header = header.strip('*').replace('**','')
            flush_bullets(); story.append(Paragraph(header, style_h3_pdf)); story.append(Spacer(1, 0.10*cm)); continue
        if line.startswith('#### '):
            header = line[5:].strip()
            header = header.strip('*').replace('**','')
            flush_bullets(); story.append(Paragraph(header, style_h3_pdf)); story.append(Spacer(1, 0.05*cm)); continue
        # Vynech úvodní automatické věty a oddělovače
        if re.match(r'^(Vynikající\.|Rozumím\.|---\s*$)', line):
            continue
        if re.match(r'^(\-|\*)\s+', line):
            bullet_buf.append(('bullet', re.sub(r'^(\-|\*)\s+', '', line))); continue
        if re.match(r'^\d+[\.)]?\s+', line):
            bullet_buf.append(('bullet', re.sub(r'^\d+[\.)]?\s+', '', line))); continue

        flush_bullets()
        txt = line.strip()
        txt = re.sub(r'\*\*(?=\S)([^\*]+?)(?<=\S)\*\*', r'<b>\1</b>', txt)
        txt = re.sub(r'_(?=\S)(.+?)(?<=\S)_', r'<i>\1</i>', txt)
        story.append(Paragraph(txt, style_normal_pdf))

    flush_bullets()


def my_page_layout_pdf(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NORMAL, 8)
    try:
        page_number_text = "Strana {}".format(doc.page)
        canvas.drawString(inch, 0.4 * inch, page_number_text)
    except Exception:
        pass
    canvas.restoreState()


def get_cell_color(value: Optional[float]) -> Optional[colors.Color]:
    if pd.isna(value):
        return None
    if value >= 105:
        return color_good_pdf
    if value < 95:
        return color_bad_pdf
    return color_avg_pdf


def generate_player_report_pdf_full(processed_data: Dict[str, Any], gemini_texts: Dict[str, str], output_filename_with_path: str):
    doc = SimpleDocTemplate(output_filename_with_path, pagesize=A4, leftMargin=0.6*inch, rightMargin=0.6*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story: list[Any] = []
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)
    p_name = processed_data.get("player_name", ""); available_page_width = A4[0] - doc.leftMargin - doc.rightMargin

    main_gauge_reduction_factor = 1.65
    section_gauge_reduction_factor = 1.1
    inter_image_padding_main = 0.1 * inch; inter_image_padding_sections = 0.1 * inch

    story.append(Paragraph("Hodnocení výkonnosti hráče", style_h1_pdf)); story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(f"<b>{p_name}</b>", style_h1_pdf)); story.append(Spacer(1, 0.3*cm))

    info_data = [
        [Paragraph("Klub:", style_normal_pdf), Paragraph(str(processed_data.get("player_club","N/A")), style_normal_pdf)],
        [Paragraph("Pozice:", style_normal_pdf), Paragraph(', '.join(processed_data.get("player_positions",["N/A"])), style_normal_pdf)],
        [Paragraph("Věk:", style_normal_pdf), Paragraph(str(int(processed_data.get("player_age",0))) if pd.notna(processed_data.get("player_age")) else "N/A", style_normal_pdf)],
        [Paragraph("Výška / Váha:", style_normal_pdf), Paragraph(f"{int(processed_data.get('player_height',0)) if pd.notna(processed_data.get('player_height')) else 'N/A'} cm / {int(processed_data.get('player_weight',0)) if pd.notna(processed_data.get('player_weight')) else 'N/A'} kg", style_normal_pdf)],
        [Paragraph("Odehrané minuty:", style_normal_pdf), Paragraph(str(processed_data.get("player_minutes",0)), style_normal_pdf)],
        [Paragraph("Datum reportu:", style_normal_pdf), Paragraph(str(processed_data.get("report_date","N/A")), style_normal_pdf)],
    ]
    info_table = Table(info_data, colWidths=[4.0*cm, available_page_width - 4.0*cm]); info_table.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(info_table); story.append(Spacer(1, 0.5*cm))

    original_main_gauge_space_each = (available_page_width - inter_image_padding_main) / 2
    img_normal_width = original_main_gauge_space_each / main_gauge_reduction_factor; img_normal_height = img_normal_width
    gauge_lg_title = "Celkový rating vs. Liga"; gauge_tp_title = "Celkový rating vs. TOP Kluby"
    safe_p_name_for_filename = p_name.replace(' ','_').replace('.','')
    gauge_lg_path = os.path.join(TEMP_IMG_DIR, f"{safe_p_name_for_filename}_overall_lg.png"); create_gauge_chart_pdf(processed_data.get("overall_score_lg"), gauge_lg_title, gauge_lg_path, size="normal")
    gauge_tp_path = os.path.join(TEMP_IMG_DIR, f"{safe_p_name_for_filename}_overall_tp.png"); create_gauge_chart_pdf(processed_data.get("overall_score_tp"), gauge_tp_title, gauge_tp_path, size="normal")
    main_gauges_table_data = [[Image(gauge_lg_path, width=img_normal_width, height=img_normal_height), Image(gauge_tp_path, width=img_normal_width, height=img_normal_height)]]
    main_gauges_table_col_widths = [ (available_page_width - inter_image_padding_main) / 2 ] * 2
    main_gauges_table = Table(main_gauges_table_data, colWidths=main_gauges_table_col_widths); main_gauges_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(main_gauges_table); story.append(Spacer(1, 0.3*cm))

    gemini_text_lang = gemini_texts.get('cs', "Analýza není k dispozici.")
    _render_gemini_to_story(story, gemini_text_lang)
    story.append(Spacer(1, 0.4*cm))

    story.append(PageBreak())
    font_size_en_h2 = style_h2_pdf.fontSize - 4
    story.append(Paragraph(f"Výkonnost v klíčových oblastech (Sekce)", style_h2_pdf))
    story.append(Spacer(1, 0.2*cm))
    sec_lg_df = processed_data.get("sec_lg_df", pd.DataFrame()); sec_tp_df = processed_data.get("sec_tp_df", pd.DataFrame())
    if not sec_lg_df.empty and "Section" in sec_lg_df.columns: sec_lg_df = sec_lg_df.set_index("Section")
    else: sec_lg_df = pd.DataFrame(columns=['Score']).rename_axis("Section")
    if not sec_tp_df.empty and "Section" in sec_tp_df.columns: sec_tp_df = sec_tp_df.set_index("Section")
    else: sec_tp_df = pd.DataFrame(columns=['Score']).rename_axis("Section")
    section_names = sec_lg_df.index.unique().union(sec_tp_df.index.unique()).tolist()

    col_sec_name_width = available_page_width * 0.45
    original_sections_gauge_space_each = (available_page_width - col_sec_name_width - inter_image_padding_sections) / 2
    col_sec_gauge_width_actual = original_sections_gauge_space_each
    img_small_width = (original_sections_gauge_space_each / section_gauge_reduction_factor) * 0.98; img_small_height = img_small_width
    section_gauge_data_for_table = []
    for i, sec_name_cs in enumerate(section_names):
        score_lg = sec_lg_df.loc[sec_name_cs, "Score"] if sec_name_cs in sec_lg_df.index and "Score" in sec_lg_df.columns else pd.NA
        score_tp = sec_tp_df.loc[sec_name_cs, "Score"] if sec_name_cs in sec_tp_df.index and "Score" in sec_tp_df.columns else pd.NA
        title_lg = "vs. AVG Liga"; title_tp = "vs. TOP Kluby"
        sec_gauge_lg_path = os.path.join(TEMP_IMG_DIR, f"{safe_p_name_for_filename}_sec_{i}_lg.png"); create_gauge_chart_pdf(score_lg, title_lg, sec_gauge_lg_path, size="small")
        sec_gauge_tp_path = os.path.join(TEMP_IMG_DIR, f"{safe_p_name_for_filename}_sec_{i}_tp.png"); create_gauge_chart_pdf(score_tp, title_tp, sec_gauge_tp_path, size="small")
        display_sec_name = f"<b>{sec_name_cs}</b>"
        section_gauge_data_for_table.append([Paragraph(display_sec_name, style_section_name_cell_pdf), Image(sec_gauge_lg_path, width=img_small_width, height=img_small_height), Image(sec_gauge_tp_path, width=img_small_width, height=img_small_height)])
    if section_gauge_data_for_table:
        sections_table_col_widths = [col_sec_name_width, col_sec_gauge_width_actual, col_sec_gauge_width_actual]
        sections_display_table = Table(section_gauge_data_for_table, colWidths=sections_table_col_widths)
        sections_display_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (1,0), (-1,-1), 'CENTER'), ('LEFTPADDING', (0,0), (0,-1), 0.1*cm), ('ALIGN', (0,0), (0,-1), 'LEFT'), ('BOTTOMPADDING', (0,0), (-1,-1), 0.3*cm)]))
        story.append(sections_display_table)

    story.append(PageBreak())
    story.append(Paragraph("Detailní pohled (Sub-sekce)", style_h2_pdf))
    story.append(Spacer(1, 0.2*cm))
    sub_lg_df_safe = processed_data.get("sub_lg_df", pd.DataFrame()); sub_tp_df_safe = processed_data.get("sub_tp_df", pd.DataFrame())
    vs_league_col_name_cs = "vs. Liga (%)"; vs_top_clubs_col_name_cs = "vs. TOP Kluby (%)"
    if sub_lg_df_safe.empty or sub_tp_df_safe.empty or "Section" not in sub_lg_df_safe.columns or "Subsection" not in sub_lg_df_safe.columns or "Score" not in sub_lg_df_safe.columns or "Section" not in sub_tp_df_safe.columns or "Subsection" not in sub_tp_df_safe.columns or "Score" not in sub_tp_df_safe.columns:
         sub_merged_df = pd.DataFrame(columns=["Section", "Subsection", vs_league_col_name_cs, vs_top_clubs_col_name_cs])
    else:
        sub_merged_df = pd.merge(sub_lg_df_safe, sub_tp_df_safe, on=["Section", "Subsection"], suffixes=('_lg', '_tp'), how="outer")
        sub_merged_df.rename(columns={'Score_lg': vs_league_col_name_cs, 'Score_tp': vs_top_clubs_col_name_cs}, inplace=True)
    header_sub_pdf = [ Paragraph("Sekce", style_table_header_pdf), Paragraph("Sub-sekce", style_table_header_pdf), Paragraph("vs. Liga", style_table_header_pdf), Paragraph("vs. TOP Kluby", style_table_header_pdf)]
    data_sub_pdf = [header_sub_pdf]
    table_styles_cmds_sub = [ ('GRID', (0,0), (-1,-1), 0.5, colors.darkgrey), ('BACKGROUND', (0,0), (-1,0), color_header_bg_pdf), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 4), ('RIGHTPADDING', (0,0), (-1,-1), 4), ('FONTNAME', (0,0), (-1,0), FONT_BOLD), ]
    last_section_name_cs = None
    for i, row_data in sub_merged_df.iterrows():
        current_section_name_cs = row_data.get("Section", ""); section_display_name_cs = current_section_name_cs if current_section_name_cs != last_section_name_cs else ""; last_section_name_cs = current_section_name_cs
        subsection_name_cs = str(row_data.get("Subsection",""))
        sec = Paragraph(section_display_name_cs, style_table_cell_left_pdf)
        sub = Paragraph(subsection_name_cs, style_table_cell_left_pdf)
        val_lg, val_tp = row_data.get(vs_league_col_name_cs), row_data.get(vs_top_clubs_col_name_cs)
        p_lg = Paragraph(f"{val_lg:.0f}%" if pd.notna(val_lg) else "N/A", style_table_cell_center_pdf); p_tp = Paragraph(f"{val_tp:.0f}%" if pd.notna(val_tp) else "N/A", style_table_cell_center_pdf)
        data_sub_pdf.append([sec, sub, p_lg, p_tp]); row_idx_pdf = len(data_sub_pdf) - 1
        color_lg_cell = get_cell_color(val_lg); 
        if color_lg_cell: table_styles_cmds_sub.append(('BACKGROUND', (2, row_idx_pdf), (2, row_idx_pdf), color_lg_cell))
        color_tp_cell = get_cell_color(val_tp); 
        if color_tp_cell: table_styles_cmds_sub.append(('BACKGROUND', (3, row_idx_pdf), (3, row_idx_pdf), color_tp_cell))
    col_width_section_sub = 4.5*cm; col_width_subsection_sub = available_page_width - (col_width_section_sub + 2.8*cm + 3.2*cm)
    sub_table_pdf = Table(data_sub_pdf, colWidths=[col_width_section_sub, col_width_subsection_sub, 2.8*cm, 3.2*cm], repeatRows=1)
    sub_table_pdf.setStyle(TableStyle(table_styles_cmds_sub)); story.append(sub_table_pdf); story.append(PageBreak())

    story.append(Paragraph("Přehled všech relevantních metrik", style_h2_pdf))
    story.append(Spacer(1, 0.2*cm))
    df_all_metrics = processed_data.get("df_all_numeric_ratings", pd.DataFrame()).copy()
    # Vylouč explicitně problematickou metriku z výpisu
    if not df_all_metrics.empty and 'Metric' in df_all_metrics.columns:
        df_all_metrics = df_all_metrics[df_all_metrics['Metric'] != 'Aerial duels per 90.1']
    bio_metrics_to_exclude = ['Age', 'Height', 'Weight', 'Market value', 'Contract expires', 'Foot', 'On loan', 'Passport country', 'Birth country', 'Position']
    df_all_metrics = df_all_metrics[~df_all_metrics['Metric'].isin(bio_metrics_to_exclude)]
    if processed_data.get('main_position') != 'GK':
        gk_metrics_to_exclude = ['Clean sheets', 'Conceded goals', 'Conceded goals per 90', 'Exits per 90', 'Prevented goals', 'Prevented goals per 90', 'Save rate', 'Shots against', 'Shots against per 90', 'xG against', 'xG against per 90']
        df_all_metrics = df_all_metrics[~df_all_metrics['Metric'].isin(gk_metrics_to_exclude)]
    header_all_pdf = [Paragraph("Metrika", style_table_header_pdf), Paragraph("Hráč", style_table_header_pdf), Paragraph("Liga Ø", style_table_header_pdf), Paragraph("TOP Kluby Ø", style_table_header_pdf), Paragraph("vs. Liga", style_table_header_pdf), Paragraph("vs. TOP Kluby", style_table_header_pdf)]
    data_all_pdf = [header_all_pdf]
    table_styles_cmds_all = [ ('GRID', (0,0), (-1,-1), 0.5, colors.darkgrey), ('BACKGROUND', (0,0), (-1,0), color_header_bg_pdf), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3), ('FONTNAME', (0,0), (-1,0), FONT_BOLD) ]

    def format_metric_value(metric_name, value):
        if pd.isna(value): return "N/A"
        if any(sub in str(metric_name).lower() for sub in ["%", " won", "accurate", "success", "conversion"]): return f"{value:.1f}"
        if value == 0: return "0"
        if abs(value) < 1 and abs(value) > 0 : return f"{value:.2f}"
        if abs(value) < 10 : return f"{value:.2f}"
        if abs(value) < 100 : return f"{value:.1f}"
        return f"{value:.0f}"

    if not df_all_metrics.empty:
        for i, row_data in df_all_metrics.iterrows():
            metric_name_val_cs = str(row_data.get("Metric","")); metric_p = Paragraph(metric_name_val_cs, style_table_cell_left_pdf)
            player_v_p = Paragraph(format_metric_value(metric_name_val_cs, row_data.get("Player")), style_table_cell_center_pdf)
            league_v_p = Paragraph(format_metric_value(metric_name_val_cs, row_data.get("League Avg")), style_table_cell_center_pdf)
            top_v_p = Paragraph(format_metric_value(metric_name_val_cs, row_data.get("TOP Clubs Avg")), style_table_cell_center_pdf)
            val_vs_lg = row_data.get("vs. League (%)"); val_vs_tp = row_data.get("vs. TOP Clubs (%)")
            # Barevné zvýraznění buněk vs. Liga a vs. TOP
            vs_lg_p = Paragraph(f"{val_vs_lg:.0f}%" if pd.notna(val_vs_lg) else "N/A", style_table_cell_center_pdf)
            vs_tp_p = Paragraph(f"{val_vs_tp:.0f}%" if pd.notna(val_vs_tp) else "N/A", style_table_cell_center_pdf)
            row_idx_pdf = len(data_all_pdf)
            data_all_pdf.append([metric_p, player_v_p, league_v_p, top_v_p, vs_lg_p, vs_tp_p])
            color_lg_cell_all = get_cell_color(val_vs_lg)
            if color_lg_cell_all: table_styles_cmds_all.append(('BACKGROUND', (4, row_idx_pdf), (4, row_idx_pdf), color_lg_cell_all))
            color_tp_cell_all = get_cell_color(val_vs_tp)
            if color_tp_cell_all: table_styles_cmds_all.append(('BACKGROUND', (5, row_idx_pdf), (5, row_idx_pdf), color_tp_cell_all))
    else:
        data_all_pdf.append([Paragraph("Žádné relevantní metriky k zobrazení.", style_table_cell_left_pdf)] + [Paragraph("", style_table_cell_center_pdf)] * 5)

    col_width_metric_name = 6.0*cm
    remaining_width_for_metrics = available_page_width - col_width_metric_name
    col_player_val = remaining_width_for_metrics * 0.14; col_league_avg = remaining_width_for_metrics * 0.14
    col_top_clubs = remaining_width_for_metrics * 0.18; col_vs_league = remaining_width_for_metrics * 0.27
    col_vs_top = remaining_width_for_metrics * 0.27
    col_widths_all_metrics = [col_width_metric_name, col_player_val, col_league_avg, col_top_clubs, col_vs_league, col_vs_top]
    all_metrics_table_pdf = Table(data_all_pdf, colWidths=col_widths_all_metrics, repeatRows=1)
    all_metrics_table_pdf.setStyle(TableStyle(table_styles_cmds_all)); story.append(all_metrics_table_pdf)
    doc.build(story, onFirstPage=my_page_layout_pdf, onLaterPages=my_page_layout_pdf)

    try:
        for item in os.listdir(TEMP_IMG_DIR):
            if item.startswith(safe_p_name_for_filename) and item.endswith(".png"):
                os.remove(os.path.join(TEMP_IMG_DIR, item))
    except Exception as e:
        print(f"Varování: Nepodařilo se smazat některé dočasné obrázky: {e}")

def generate_all_players_reports_web_detailed(excel_file_path: str, output_dir: str = "temp_pdf_reports") -> tuple[List[str], List[Dict[str, str]]]:
    """Stejné jako generate_all_players_reports_web, ale vrací i důvody pro přeskočené hráče.

    Returns: (generated_files, skipped_info)
      - generated_files: List cest k PDF
      - skipped_info: List záznamů {"player": jméno, "reason": důvod}
    """
    generated_files: List[str] = []
    skipped: List[Dict[str, str]] = []
    try:
        df_player_data_full = pd.read_excel(excel_file_path, engine="openpyxl")
        df_player_data_full = preprocess_dataframe(df_player_data_full)

        df_combined_avg = _load_combined_avg_dataframe()
        if df_combined_avg.empty:
            return [], [{"player": "ALL", "reason": "Prázdná nebo nedostupná AVG data."}]

        os.makedirs(output_dir, exist_ok=True)

        if "Player" not in df_player_data_full.columns or df_player_data_full.empty:
            return [], [{"player": "ALL", "reason": "Vstupní XLSX neobsahuje sloupec 'Player' nebo je prázdný."}]

        player_list = (
            df_player_data_full["Player"].dropna().astype(str).map(lambda x: x.strip()).unique().tolist()
        )

        for player_name in player_list:
            try:
                processed_data: Optional[Dict[str, Any]] = extract_and_process_data_for_pdf(
                    player_name, df_player_data_full, df_combined_avg
                )
                if not processed_data:
                    # Pokusíme se odhadnout důvod z typických situací
                    try:
                        rows_p = df_player_data_full[df_player_data_full["Player"] == player_name]
                        if not rows_p.empty:
                            minutes = pd.to_numeric(rows_p.get("Minutes played"), errors="coerce").fillna(0).max()
                            min_required = int(MIN_MINUTES)
                            if minutes < min_required:
                                skipped.append({"player": player_name, "reason": f"Nedostatek minut: {int(minutes)} < {min_required}."})
                                continue
                            pos = str(rows_p.get(COL_POS, pd.Series(["N/A"])) .iloc[0])
                            skipped.append({"player": player_name, "reason": f"Nedostatek průměrných dat pro pozici: {pos}."})
                            continue
                        skipped.append({"player": player_name, "reason": "Hráč nenalezen po předzpracování."})
                        continue
                    except Exception:
                        skipped.append({"player": player_name, "reason": "Neznámý důvod (výjimka při diagnostice)."})
                        continue

                gemini_available: bool = bool(_gemini_pdf_renderer_available)
                if gemini_available:
                    gemini_texts: Dict[str, str] = get_gemini_analysis_for_pdf(processed_data)
                else:
                    gemini_texts = {
                        "cs": "Textová analýza Gemini AI není k dispozici.",
                        "en": "Gemini AI textual analysis is not available.",
                    }

                current_date_str = datetime.date.today().strftime("%Y-%m-%d")
                safe_player_name = processed_data.get("player_name", player_name).replace(" ", "_").replace(".", "")
                filename = f"{safe_player_name}_{current_date_str}_performance_report.pdf"
                output_path = os.path.join(output_dir, filename)

                generate_player_report_pdf_full(processed_data, gemini_texts, output_path)

                if os.path.exists(output_path):
                    generated_files.append(output_path)
                else:
                    skipped.append({"player": player_name, "reason": "PDF soubor nebyl vytvořen."})
            except Exception as e:
                print(f"Chyba při generování reportu pro hráče {player_name}: {e}")
                traceback.print_exc()
                skipped.append({"player": player_name, "reason": f"Výjimka: {e}"})
                continue

        return generated_files, skipped

    except Exception as e:
        print(f"Kritická chyba při dávkovém generování PDF: {e}")
        traceback.print_exc()
        return [], [{"player": "ALL", "reason": f"Kritická chyba: {e}"}]


def generate_all_players_reports_web(excel_file_path: str, output_dir: str = "temp_pdf_reports") -> List[str]:
    """Zachovaná původní signatura pro kompatibilitu se stávajícím kódem UI."""
    files, _skipped = generate_all_players_reports_web_detailed(excel_file_path, output_dir)
    return files


