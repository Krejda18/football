# player_analysis.py
import streamlit as st
import pandas as pd
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List

from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part, GenerationConfig

from core_logic import (
    compute_loose_ball_duels, compute_effective_passing_index,
    rating_series, weighted_score, logic_flat_df, breakdown_scores, build_prompt,
    get_positions_for_avg_filter
)

# --- Konfigurace ---
DATA_DIR = Path("/Users/krejda/Documents/Python/Football/Player_Rating/Data")
AVG_DATA_DIR = Path("/Users/krejda/Documents/Python/Football/Player_Rating/AVG - hodnoty")
LOGIC_JSON = Path("/Users/krejda/Documents/Python/Football/metric_logic.json")
LOGO_DIR = Path("./logos")
TOP_CLUBS = ["Slavia Praha", "Sparta Praha", "Viktoria Plzeň"]
COL_POS = "Converted Position"
MIN_MINUTES = 700
SERVICE_ACCOUNT_JSON = "service-account-key.json"
PROJECT_ID = "performance-445519"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-pro"

# --- Cachované funkce ---
@st.cache_data
def load_and_process_file(file_path: Path) -> pd.DataFrame:
    print(f"--- CACHE MISS: Načítám a zpracovávám soubor: {file_path.name} ---")
    df = pd.read_excel(file_path, engine="openpyxl")
    df = df[df["Converted Position"] != 'GK']
    df = compute_loose_ball_duels(df)
    df = compute_effective_passing_index(df)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

@st.cache_data
def get_logic_definition() -> dict:
    with open(LOGIC_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def initialize_gemini() -> tuple[GenerativeModel | None, bool]:
    try:
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        model = GenerativeModel(MODEL_NAME)
        print("--- GEMINI MODEL ÚSPĚŠNĚ INICIALIZOVÁN ---")
        return model, True
    except Exception as e:
        st.warning(f"Nepodařilo se inicializovat Gemini AI: {e}")
        return None, False

@st.cache_data
def load_all_player_data() -> pd.DataFrame:
    all_player_dfs = []
    for file_path in sorted(Path(DATA_DIR).glob("*.xlsx")):
        df = load_and_process_file(file_path)
        df['League'] = file_path.stem
        all_player_dfs.append(df)
    if not all_player_dfs:
        return pd.DataFrame()
    combined_df = pd.concat(all_player_dfs, ignore_index=True)
    return combined_df[combined_df["Minutes played"] >= MIN_MINUTES]

@st.cache_data
def calculate_ratings_for_all_players(all_players_df: pd.DataFrame, all_avg_df: pd.DataFrame) -> pd.DataFrame:
    logic_data = get_logic_definition()
    results = []
    position_groups = all_players_df[COL_POS].dropna().unique()
    avg_calcs = {}
    for pos in position_groups:
        positions_to_filter = get_positions_for_avg_filter(pos)
        avg_df_pos_filtered = all_avg_df[all_avg_df[COL_POS].isin(positions_to_filter)]
        if not avg_df_pos_filtered.empty:
            lg_avg = avg_df_pos_filtered.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)
            top_clubs_df = avg_df_pos_filtered[avg_df_pos_filtered["Team"].isin(TOP_CLUBS)]
            tp_avg = pd.Series(dtype='float64') if top_clubs_df.empty else top_clubs_df.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)
            avg_calcs[pos] = {'lg': lg_avg, 'tp': tp_avg}

    for _, player_row in all_players_df.iterrows():
        main_position = player_row[COL_POS]
        if main_position in avg_calcs:
            p_avg = player_row
            lg_avg, tp_avg = avg_calcs[main_position]['lg'], avg_calcs[main_position]['tp']
            rat_lg, rat_tp = rating_series(p_avg, lg_avg), rating_series(p_avg, tp_avg)
            logic_df = logic_flat_df([main_position], logic_data)
            logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_lg.index]
            score_lg = weighted_score(rat_lg[logic_metrics_in_data], logic_df)
            score_tp = weighted_score(rat_tp[logic_metrics_in_data], logic_df)
            results.append({
                'Player': player_row['Player'], 'Team': player_row['Team'],
                'League': player_row['League'], 'Position': main_position,
                'Age': player_row['Age'], 'Height': player_row['Height'],
                'Market value': player_row['Market value'],
                'Rating vs Liga': score_lg, 'Rating vs TOP Kluby': score_tp
            })
    
    final_df = pd.DataFrame(results)
    
    # --- ZMĚNA ZDE: Zaokrouhlení ratingů na celá čísla ---
    rating_cols = ['Rating vs Liga', 'Rating vs TOP Kluby']
    final_df[rating_cols] = final_df[rating_cols].round(0)

    column_order = ['Player', 'Team', 'League', 'Position', 'Age', 'Height', 'Market value', 'Rating vs Liga', 'Rating vs TOP Kluby']
    return final_df[column_order]

# V souboru player_analysis.py

def analyze_player(player_name: str, player_df: pd.DataFrame, avg_df: pd.DataFrame) -> Dict[str, Any]:
    gemini_model, gemini_available = initialize_gemini()
    logic_data = get_logic_definition()
    
    player_rows = player_df[player_df["Player"] == player_name]
    p_avg = player_rows.mean(numeric_only=True)
    main_position = player_rows[COL_POS].iloc[0]

    positions_to_filter = get_positions_for_avg_filter(main_position)
    avg_df_pos_filtered = avg_df[avg_df[COL_POS].isin(positions_to_filter)]
    
    if avg_df_pos_filtered.empty:
        st.warning(f"Pro pozice '{', '.join(positions_to_filter)}' nebyla nalezena žádná srovnávací data.")
        avg_df_pos_filtered = avg_df

    lg_avg = avg_df_pos_filtered.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)
    
    top_clubs_df = avg_df_pos_filtered[avg_df_pos_filtered["Team"].isin(TOP_CLUBS)]
    if top_clubs_df.empty:
        tp_avg = pd.Series(dtype='float64')
    else:
        tp_avg = top_clubs_df.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)

    rat_lg = rating_series(p_avg, lg_avg)
    rat_tp = rating_series(p_avg, tp_avg)
    
    logic_df = logic_flat_df([main_position], logic_data)
    logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_lg.index]

    score_lg = weighted_score(rat_lg[logic_metrics_in_data], logic_df)
    score_tp = weighted_score(rat_tp[logic_metrics_in_data], logic_df)

    sec_lg, sub_lg = breakdown_scores(rat_lg, main_position, logic_data)
    sec_tp, sub_tp = breakdown_scores(rat_tp, main_position, logic_data)
    
    sec_tbl = sec_lg.rename(columns={"Score": "vs. League"}).merge(sec_tp.rename(columns={"Score": "vs. TOP 3"}), on="Section", how="outer")
    sub_tbl = sub_lg.rename(columns={"Score": "vs. League"}).merge(sub_tp.rename(columns={"Score": "vs. TOP 3"}), on=["Section", "Subsection"], how="outer")
    
    GK_METRICS_TO_EXCLUDE = [
        'Clean sheets', 'Conceded goals', 'Conceded goals per 90', 'Exits per 90',
        'Prevented goals', 'Prevented goals per 90', 'Save rate',
        'Shots against', 'Shots against per 90', 'xG against', 'xG against per 90',
        'Back passes received as GK per 90'
    ]
    common_metrics = p_avg.index.intersection(lg_avg.index)
    metrics_to_display = [m for m in common_metrics if m not in GK_METRICS_TO_EXCLUDE]
    rows = []
    for m in sorted(metrics_to_display):
        val_lg = (p_avg.get(m, 0) / lg_avg.get(m, 0) * 100) if lg_avg.get(m, 0) != 0 else pd.NA
        val_tp = (p_avg.get(m, 0) / tp_avg.get(m, 0) * 100) if tp_avg.get(m, 0) != 0 else pd.NA
        rows.append({"Metric": m, "Hráč": p_avg.get(m), "Liga Ø": lg_avg.get(m), "TOP Kluby Ø": tp_avg.get(m, pd.NA), "vs. League": val_lg, "vs. TOP 3": val_tp})
    all_metrics_tbl = pd.DataFrame(rows)
    
    analysis_text = "AI analýza nebyla vygenerována." # ... kód Gemini ...

    player_row = player_rows.iloc[0]
    player_club = player_row.get("Team", "N/A")
    logo_path = LOGO_DIR / f"{player_club}.png"
    if not logo_path.is_file(): logo_path = None

    # --- ZMĚNA ZDE: Odstranění logiky pro logo ---
    full_header_block = f"""
    # {player_name}
    ### 🧾 Základní informace
    **Klub:** {player_club}<br>
    **Pozice:** {main_position}<br>
    **Věk:** {int(player_row.get('Age', 0))}<br>
    **Výška:** {int(player_row.get('Height', 0))} cm<br>
    **Minuty:** {int(player_row.get('Minutes played', 0))}
    """
    # --- KONEC ZMĚNY ---

    return {
        "full_header_block": full_header_block,
        "score_lg": score_lg, "score_tp": score_tp,
        "sec_tbl": sec_tbl, "sub_tbl": sub_tbl, "all_metrics": all_metrics_tbl,
        "analysis": analysis_text, "gemini_available": gemini_available,
    }