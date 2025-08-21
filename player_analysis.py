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
    get_positions_for_avg_filter, build_ai_scout_prompt
)

# --- Konfigurace ---
DATA_DIR = Path("./Data")
AVG_DATA_DIR = Path("./AVG - hodnoty")
LOGIC_JSON = Path("metric_logic.json")
LOGO_DIR = Path("./logos")
TOP_CLUBS = ["Slavia Praha", "Sparta Praha", "Viktoria Plzeň"]
COL_POS = "Converted Position"
MIN_MINUTES = 500
SERVICE_ACCOUNT_JSON = "inside-data-story-af484f6c4b69.json"
PROJECT_ID = "inside-data-story"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"

# --- Cachované funkce ---
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
    """
    Inicializuje model Gemini.
    Pokusí se načíst klíč ze Streamlit Secrets (pro nasazení na cloudu).
    Pokud selže, pokusí se načíst klíč z lokálního souboru (pro lokální vývoj).
    """
    creds = None
    try:
        # Pokus č. 1: Načtení ze Streamlit Secrets (pro Cloud)
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        print("--- ÚSPĚCH: Gemini inicializován ze Streamlit Secrets. ---")
    except Exception:
        # Pokus č. 2: Načtení z lokálního souboru (pro lokální vývoj)
        print(f"--- Secrets nenalezeny, pokouším se načíst lokální soubor: {SERVICE_ACCOUNT_JSON} ---")
        try:
            creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
            print("--- ÚSPĚCH: Gemini inicializován z lokálního souboru. ---")
        except Exception as e:
            st.warning(f"Nepodařilo se inicializovat Gemini. Klíč nenalezen ani v Secrets, ani lokálně. Chyba: {e}")
            return None, False

    # Společná inicializace Vertex AI po úspěšném načtení klíče
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        model = GenerativeModel(MODEL_NAME)
        return model, True
    except Exception as e:
        st.warning(f"Podařilo se načíst klíč, ale selhala inicializace Vertex AI: {e}")
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
    
    analysis_text = "AI analýza není dostupná."
    if gemini_available and gemini_model:
        try:
            # --- ZMĚNA ZDE: Volání nové verze `build_prompt` se správnými daty (tabulkami) ---
            prompt = build_prompt(player_name, [main_position], sec_tbl, sub_tbl, all_metrics_tbl)
            msg = Content(role="user", parts=[Part.from_text(prompt)])
            config = GenerationConfig(max_output_tokens=10000, temperature=0.5, top_k=30)
            response = gemini_model.generate_content([msg], generation_config=config)
            analysis_text = response.text
        except Exception as e:
            analysis_text = f"Došlo k chybě při generování AI analýzy: {e}"

    player_row = player_rows.iloc[0]
    player_club = player_row.get("Team", "N/A")
    logo_path = LOGO_DIR / f"{player_club}.png"
    if not logo_path.is_file(): logo_path = None
    
    full_header_block = f"""
    # {player_name}
    ### 🧾 Základní informace
    **Klub:** {player_club}<br>
    **Pozice:** {main_position}<br>
    **Věk:** {int(player_row.get('Age', 0))}<br>
    **Výška:** {int(player_row.get('Height', 0))} cm<br>
    **Minuty:** {int(player_row.get('Minutes played', 0))}
    """
    
    return {
        "full_header_block": full_header_block,
        "logo_path": str(logo_path) if logo_path else None,
        "score_lg": score_lg, "score_tp": score_tp,
        "sec_tbl": sec_tbl, "sub_tbl": sub_tbl,
        "all_metrics": all_metrics_tbl,
        "analysis": analysis_text,
        "gemini_available": gemini_available,
    }

# V souboru player_analysis.py

@st.cache_data
def calculate_all_player_metrics_and_ratings(all_players_df: pd.DataFrame, all_avg_df: pd.DataFrame) -> pd.DataFrame:
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
            p_avg, lg_avg, tp_avg = player_row, avg_calcs[main_position]['lg'], avg_calcs[main_position]['tp']
            rat_lg, rat_tp = rating_series(p_avg, lg_avg), rating_series(p_avg, tp_avg)
            logic_df = logic_flat_df([main_position], logic_data)
            logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_lg.index]
            score_lg, score_tp = weighted_score(rat_lg[logic_metrics_in_data], logic_df), weighted_score(rat_tp[logic_metrics_in_data], logic_df)
            results.append({
                'Player': player_row['Player'], 'Team': player_row['Team'],
                'League': player_row['League'], 'Position': main_position,
                'Age': player_row['Age'], 'Height': player_row['Height'],
                'Market value': player_row.get('Market value'),
                'Foot': player_row.get('Foot'),
                'Minutes': player_row['Minutes played'],
                'Rating vs Liga': score_lg, 'Rating vs TOP Kluby': score_tp
            })
    
    final_df = pd.DataFrame(results)
    rating_cols = ['Rating vs Liga', 'Rating vs TOP Kluby']
    final_df[rating_cols] = final_df[rating_cols].round(0)
    column_order = ['Player', 'Team', 'League', 'Position', 'Age', 'Height', 'Foot', "Minutes", 'Market value', 'Rating vs Liga', 'Rating vs TOP Kluby']
    return final_df.reindex(columns=column_order)

@st.cache_data
def enrich_data_for_ai_scout(all_players_df: pd.DataFrame, all_avg_df: pd.DataFrame) -> pd.DataFrame:
    logic_data = get_logic_definition()
    enriched_results = []
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
            p_avg, lg_avg, tp_avg = player_row, avg_calcs[main_position]['lg'], avg_calcs[main_position]['tp']
            rat_lg, rat_tp = rating_series(p_avg, lg_avg), rating_series(p_avg, tp_avg)
            logic_df = logic_flat_df([main_position], logic_data)
            logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_lg.index]
            score_lg = weighted_score(rat_lg[logic_metrics_in_data], logic_df)
            player_data = player_row.to_dict()
            player_data['Rating vs Liga'] = score_lg
            sec_lg, sub_lg = breakdown_scores(rat_lg, main_position, logic_data)
            sec_tp, sub_tp = breakdown_scores(rat_tp, main_position, logic_data)
            sec_ratings = sec_lg.rename(columns={"Score": "lg"}).merge(sec_tp.rename(columns={"Score": "tp"}), on="Section", how="outer").to_dict('records')
            sub_ratings = sub_lg.rename(columns={"Score": "lg"}).merge(sub_tp.rename(columns={"Score": "tp"}), on=["Section", "Subsection"], how="outer").to_dict('records')
            player_data['sections'] = sec_ratings
            player_data['subsections'] = sub_ratings
            enriched_results.append(player_data)
    
    return pd.DataFrame(enriched_results)


def prepare_data_for_ai_scout(ratings_df: pd.DataFrame) -> str:
    """Převede Obohacená data hráčů na textový formát pro AI."""
    player_strings = []
    for _, row in ratings_df.iterrows():
        base_info = (
            f"Hráč: {row['Player']}, Tým: {row['Team']}, Soutěž: {row['League']}, Pozice: {row['Converted Position']}, "
            f"Věk: {row['Age']:.0f}, Noha: {row.get('Foot', 'N/A')}"
        )
        sec_info = ", ".join([ f"{s['Section']}: {s.get('lg', 0):.0f}/{s.get('tp', 0):.0f}" for s in row.get('sections', []) if pd.notna(s.get('lg')) ])
        sub_info = ", ".join([ f"{s['Subsection']}: {s.get('lg', 0):.0f}/{s.get('tp', 0):.0f}" for s in row.get('subsections', []) if pd.notna(s.get('lg')) ])
        
        player_strings.append(f"{base_info}\n  - Sekce -> {sec_info}\n  - Podsekce -> {sub_info}\n---")

    return "\n".join(player_strings)


def run_ai_scout(user_needs: str) -> str:
    """Spustí celý proces AI skautingu s obohacenými daty."""
    gemini_model, gemini_available = initialize_gemini()
    if not gemini_available:
        return "Chyba: Služba Gemini AI není dostupná."

    # Načtení dat a jejich OBOHACENÍ pro AI
    all_players_df = load_all_player_data()
    avg_files = list(Path(AVG_DATA_DIR).glob("*.xlsx"))
    all_avg_dfs = [load_and_process_file(file) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]
    
    # Použijeme novou funkci pro obohacení dat
    enriched_df = enrich_data_for_ai_scout(all_players_df, avg_df_filtered)

    if enriched_df.empty:
        return "Nenalezena žádná data hráčů pro analýzu."

    # Omezení počtu hráčů pro AI (pošleme TOP 150)
    final_candidates_df = enriched_df.nlargest(500, 'Rating vs Liga')

    # Příprava dat a spuštění AI
    players_data_string = prepare_data_for_ai_scout(final_candidates_df)
    prompt = build_ai_scout_prompt(user_needs, players_data_string)
    
    try:
        msg = Content(role="user", parts=[Part.from_text(prompt)])
        config = GenerationConfig(max_output_tokens=8192, temperature=0.6)
        response = gemini_model.generate_content([msg], generation_config=config)
        return response.text
    except Exception as e:
        return f"Došlo k chybě při komunikaci s Gemini AI: {e}"
    

def get_custom_comparison(player_series: pd.Series, main_position: str, custom_positions: list, all_avg_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Vypočítá srovnání jednoho hráče vůči průměru vybrané skupiny pozic.
    """
    if not custom_positions:
        return {}

    logic_data = get_logic_definition()
    
    # 1. Vytvoříme průměr pro vlastní skupinu
    custom_avg_df = all_avg_df[all_avg_df[COL_POS].isin(custom_positions)]
    if custom_avg_df.empty:
        return {"error": "Pro vybrané pozice nebyla nalezena žádná srovnávací data."}
        
    custom_avg = custom_avg_df.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)

    # 2. Provedeme srovnání (rating, sekce, podsekce)
    rat_custom = rating_series(player_series, custom_avg)
    logic_df = logic_flat_df([main_position], logic_data)
    logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_custom.index]
    
    score_custom = weighted_score(rat_custom[logic_metrics_in_data], logic_df)
    sec_custom, sub_custom = breakdown_scores(rat_custom, main_position, logic_data)
    
    # Přejmenujeme sloupce pro přehlednost
    sec_tbl = sec_custom.rename(columns={"Score": "vs. Vlastní výběr"})
    sub_tbl = sub_custom.rename(columns={"Score": "vs. Vlastní výběr"})

    return {
        "score": score_custom,
        "sec_tbl": sec_tbl,
        "sub_tbl": sub_tbl
    }


def get_player_comparison_data(player1_name: str, player2_name: str, player_df: pd.DataFrame, avg_df: pd.DataFrame) -> Dict[str, Any]:
    """Připraví data pro srovnání dvou hráčů."""
    
    result1 = analyze_player(player1_name, player_df, avg_df)
    result2 = analyze_player(player2_name, player_df, avg_df)

    # Získáme zkrácená jména pro nadpisy sloupců
    p1_name_short = player1_name.split(' ')[-1]
    p2_name_short = player2_name.split(' ')[-1]
    
    # Přejmenujeme sloupce s ratingy, aby se daly spojit
    sec_tbl1 = result1["sec_tbl"].rename(columns={"vs. League": f"{p1_name_short} vs. Liga", "vs. TOP 3": f"{p1_name_short} vs. TOP 3"})
    sec_tbl2 = result2["sec_tbl"].rename(columns={"vs. League": f"{p2_name_short} vs. Liga", "vs. TOP 3": f"{p2_name_short} vs. TOP 3"})
    
    sub_tbl1 = result1["sub_tbl"].rename(columns={"vs. League": f"{p1_name_short} vs. Liga", "vs. TOP 3": f"{p1_name_short} vs. TOP 3"})
    sub_tbl2 = result2["sub_tbl"].rename(columns={"vs. League": f"{p2_name_short} vs. Liga", "vs. TOP 3": f"{p2_name_short} vs. TOP 3"})

    # Přejmenujeme sloupce s absolutními hodnotami
    all_m_tbl1 = result1["all_metrics"][['Metric', 'Hráč']].rename(columns={"Hráč": p1_name_short})
    all_m_tbl2 = result2["all_metrics"][['Metric', 'Hráč']].rename(columns={"Hráč": p2_name_short})

    # Spojíme tabulky
    comparison_sec = pd.merge(sec_tbl1, sec_tbl2, on="Section")
    comparison_sub = pd.merge(sub_tbl1, sub_tbl2, on=["Section", "Subsection"])
    comparison_all = pd.merge(all_m_tbl1, all_m_tbl2, on="Metric")
    
    return {
        "result1": result1,
        "result2": result2,
        "p1_name_short": p1_name_short,
        "p2_name_short": p2_name_short,
        "comparison_sec": comparison_sec,
        "comparison_sub": comparison_sub,
        "comparison_all": comparison_all
    }
