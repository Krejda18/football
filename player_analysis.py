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
import os
from core_logic import (
    compute_loose_ball_duels, compute_effective_passing_index,
    rating_series, weighted_score, logic_flat_df, breakdown_scores, build_prompt,
    get_positions_for_avg_filter, build_ai_scout_prompt, build_head_to_head_prompt
)

# --- Konfigurace ---
DATA_DIR = Path("./Data_Parquet")
AVG_DATA_DIR = Path("./AVG_Parquet")
LOGIC_JSON = Path("metric_logic.json")
LOGO_DIR = Path("./logos")
TOP_CLUBS = ["Slavia Praha", "Sparta Praha", "Viktoria Plzeň"]
COL_POS = "Converted Position"
MIN_MINUTES = 500



# --- Konfigurace pro Google Cloud & Gemini ---
PROJECT_ID = "inside-data-story"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"  # Opravený název modelu

# Názvy pro tajné klíče v různých prostředích
ENV_SECRET_NAME = "GCP_SA_JSON"
STREAMLIT_SECRET_NAME = "gcp_service_account"
LOCAL_SECRET_PATH = "inside-data-story-af484f6c4b69.json"

# Metriky, které nechci v přehledu "Všechny metriky" a v H2H srovnání metrik
EXCLUDE_FROM_ALL_METRICS = {
    "Market value",
    "Matches played",
    "Minutes played",
    "Weight",
    "Height",
    "Age"
}

@st.cache_resource
def initialize_gemini() -> tuple[GenerativeModel | None, bool]:
    creds = None
    secret_info = None
    env_val = os.environ.get(ENV_SECRET_NAME)
    if env_val:
        try:
            secret_info = json.loads(env_val)
            print("--- INFO: Nalezen klíč v proměnné prostředí (ENV). ---")
        except json.JSONDecodeError as e:
            st.error(f"Chyba při parsování JSON z proměnné prostředí '{ENV_SECRET_NAME}': {e}")
            return None, False
    elif os.path.exists(LOCAL_SECRET_PATH):
        try:
            with open(LOCAL_SECRET_PATH) as f:
                secret_info = json.load(f)
            print(f"--- INFO: Nalezen klíč v lokálním souboru '{LOCAL_SECRET_PATH}'. ---")
        except Exception as e:
            st.error(f"Chyba při čtení lokálního souboru s klíčem: {e}")
            return None, False
    elif hasattr(st, 'secrets') and STREAMLIT_SECRET_NAME in st.secrets:
        secret_info = dict(st.secrets[STREAMLIT_SECRET_NAME])
        print("--- INFO: Nalezen klíč ve Streamlit Secrets. ---")
    if secret_info:
        try:
            creds = service_account.Credentials.from_service_account_info(secret_info)
        except Exception as e:
            st.error(f"Chyba při vytváření přihlašovacích údajů z nalezeného klíče: {e}")
            return None, False
    else:
        st.error(
            "Chybí přihlašovací údaje pro Google Cloud! Zkontrolujte nastavení pro vaše prostředí:\n\n"
            f"- **Pro Google Cloud Run:** Ujistěte se, že máte nastavený secret jako proměnnou prostředí s názvem `{ENV_SECRET_NAME}`.\n"
            f"- **Pro lokální spuštění:** Ujistěte se, že v kořenovém adresáři existuje soubor `{LOCAL_SECRET_PATH}`.\n"
            f"- **Pro Streamlit Cloud:** Ujistěte se, že máte v nastavení aplikace přidaný secret s názvem `{STREAMLIT_SECRET_NAME}`."
        )
        return None, False
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        model = GenerativeModel(MODEL_NAME)
        print("--- INFO: Vertex AI úspěšně inicializováno. ---")
        return model, True
    except Exception as e:
        st.warning(f"Klíč byl načten, ale selhala inicializace Vertex AI: {e}")
        return None, False

# --- Cachované funkce ---
# <<< ZMĚNA ZDE: Funkce nyní čte Parquet soubory, což je mnohem rychlejší >>>
def load_and_process_file(file_path: Path) -> pd.DataFrame:
    print(f"--- CACHE MISS: Načítám a zpracovávám soubor: {file_path.name} ---")
    df = pd.read_parquet(file_path, engine="pyarrow")
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

@st.cache_data
def load_all_player_data() -> pd.DataFrame:
    all_player_dfs = []
    # <<< ZMĚNA ZDE: Hledáme soubory s koncovkou .parquet >>>
    for file_path in sorted(Path(DATA_DIR).glob("*.parquet")):
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

    metrics_to_display = [
        m for m in common_metrics
        if m not in GK_METRICS_TO_EXCLUDE
        and m not in EXCLUDE_FROM_ALL_METRICS
    ]

    rows = []
    for m in sorted(metrics_to_display):
        val_lg = (p_avg.get(m, 0) / lg_avg.get(m, 0) * 100) if lg_avg.get(m, 0) != 0 else pd.NA
        val_tp = (p_avg.get(m, 0) / tp_avg.get(m, 0) * 100) if tp_avg.get(m, 0) != 0 else pd.NA
        rows.append({
            "Metric": m,
            "Hráč": p_avg.get(m),
            "Liga Ø": lg_avg.get(m),
            "TOP Kluby Ø": tp_avg.get(m, pd.NA),
            "vs. League": val_lg,
            "vs. TOP 3": val_tp
        })
    all_metrics_tbl = pd.DataFrame(rows)
    
    analysis_text = "AI analýza není dostupná."
    if gemini_available and gemini_model:
        try:
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
    **Váha:** {int(player_row.get('Weight', 0))} kg<br>
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

# Ostatní funkce zůstávají beze změny...
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
    gemini_model, gemini_available = initialize_gemini()
    if not gemini_available:
        return "Chyba: Služba Gemini AI není dostupná."

    all_players_df = load_all_player_data()
    avg_files = list(Path(AVG_DATA_DIR).glob("*.parquet"))
    all_avg_dfs = [load_and_process_file(file) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]
    
    enriched_df = enrich_data_for_ai_scout(all_players_df, avg_df_filtered)

    if enriched_df.empty:
        return "Nenalezena žádná data hráčů pro analýzu."

    final_candidates_df = enriched_df.nlargest(500, 'Rating vs Liga')

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
    if not custom_positions:
        return {}

    logic_data = get_logic_definition()
    
    custom_avg_df = all_avg_df[all_avg_df[COL_POS].isin(custom_positions)]
    if custom_avg_df.empty:
        return {"error": "Pro vybrané pozice nebyla nalezena žádná srovnávací data."}
        
    custom_avg = custom_avg_df.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)

    rat_custom = rating_series(player_series, custom_avg)
    logic_df = logic_flat_df([main_position], logic_data)
    logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_custom.index]
    
    score_custom = weighted_score(rat_custom[logic_metrics_in_data], logic_df)
    sec_custom, sub_custom = breakdown_scores(rat_custom, main_position, logic_data)
    
    sec_tbl = sec_custom.rename(columns={"Score": "vs. Vlastní výběr"})
    sub_tbl = sub_custom.rename(columns={"Score": "vs. Vlastní výběr"})

    return {
        "score": score_custom,
        "sec_tbl": sec_tbl,
        "sub_tbl": sub_tbl
    }


def get_player_comparison_data(player1_name: str, player2_name: str, player_df: pd.DataFrame, avg_df: pd.DataFrame) -> Dict[str, Any]:
    result1 = analyze_player(player1_name, player_df, avg_df)
    result2 = analyze_player(player2_name, player_df, avg_df)

    p1_name_short = player1_name.split(' ')[-1]
    p2_name_short = player2_name.split(' ')[-1]
    
    sec_tbl1 = result1["sec_tbl"].rename(columns={"vs. League": f"{p1_name_short} vs. Liga", "vs. TOP 3": f"{p1_name_short} vs. TOP 3"})
    sec_tbl2 = result2["sec_tbl"].rename(columns={"vs. League": f"{p2_name_short} vs. Liga", "vs. TOP 3": f"{p2_name_short} vs. TOP 3"})
    
    sub_tbl1 = result1["sub_tbl"].rename(columns={"vs. League": f"{p1_name_short} vs. Liga", "vs. TOP 3": f"{p1_name_short} vs. TOP 3"})
    sub_tbl2 = result2["sub_tbl"].rename(columns={"vs. League": f"{p2_name_short} vs. Liga", "vs. TOP 3": f"{p2_name_short} vs. TOP 3"})

    all_m_tbl1 = result1["all_metrics"][['Metric', 'Hráč']].rename(columns={"Hráč": p1_name_short})
    all_m_tbl2 = result2["all_metrics"][['Metric', 'Hráč']].rename(columns={"Hráč": p2_name_short})

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

def analyze_head_to_head(player1_name: str, player2_name: str, player_df: pd.DataFrame, avg_df: pd.DataFrame) -> str:
    r1 = analyze_player(player1_name, player_df, avg_df)
    r2 = analyze_player(player2_name, player_df, avg_df)

    def _header_for(name: str) -> dict:
        rows = player_df[player_df["Player"] == name]
        if rows.empty:
            return {"Team": "N/A", "Position": "N/A", "Age": 0, "Height": 0, "Minutes": 0}
        row0 = rows.iloc[0]
        return {
            "Team": row0.get("Team", "N/A"),
            "Position": row0.get(COL_POS, "N/A"),
            "Age": int(row0.get("Age", 0) or 0),
            "Height": int(row0.get("Height", 0) or 0),
            "Minutes": int(row0.get("Minutes played", 0) or 0),
        }

    h1, h2 = _header_for(player1_name), _header_for(player2_name)

    s1 = r1["sec_tbl"].rename(columns={"vs. League": "P1 vs. Liga", "vs. TOP 3": "P1 vs. TOP 3"})
    s2 = r2["sec_tbl"].rename(columns={"vs. League": "P2 vs. Liga", "vs. TOP 3": "P2 vs. TOP 3"})
    sec_delta = s1.merge(s2, on="Section", how="inner")
    if {"P1 vs. Liga", "P2 vs. Liga"}.issubset(sec_delta.columns):
        sec_delta["Δ vs. Liga"] = sec_delta["P1 vs. Liga"] - sec_delta["P2 vs. Liga"]
    if {"P1 vs. TOP 3", "P2 vs. TOP 3"}.issubset(sec_delta.columns):
        sec_delta["Δ vs. TOP 3"] = sec_delta["P1 vs. TOP 3"] - sec_delta["P2 vs. TOP 3"]

    u1 = r1["sub_tbl"].rename(columns={"vs. League": "P1 vs. Liga", "vs. TOP 3": "P1 vs. TOP 3"})
    u2 = r2["sub_tbl"].rename(columns={"vs. League": "P2 vs. Liga", "vs. TOP 3": "P2 vs. TOP 3"})
    sub_delta = u1.merge(u2, on=["Section", "Subsection"], how="inner")
    if {"P1 vs. Liga", "P2 vs. Liga"}.issubset(sub_delta.columns):
        sub_delta["Δ vs. Liga"] = sub_delta["P1 vs. Liga"] - sub_delta["P2 vs. Liga"]
    if {"P1 vs. TOP 3", "P2 vs. TOP 3"}.issubset(sub_delta.columns):
        sub_delta["Δ vs. TOP 3"] = sub_delta["P1 vs. TOP 3"] - sub_delta["P2 vs. TOP 3"]

    all1 = r1["all_metrics"][["Metric", "vs. League"]].rename(columns={"vs. League": "P1 vs. League"})
    all2 = r2["all_metrics"][["Metric", "vs. League"]].rename(columns={"vs. League": "P2 vs. League"})
    top_metrics = all1.merge(all2, on="Metric", how="inner")
    top_metrics["P1 vs. League"] = pd.to_numeric(top_metrics["P1 vs. League"], errors="coerce")
    top_metrics["P2 vs. League"] = pd.to_numeric(top_metrics["P2 vs. League"], errors="coerce")
    top_metrics["DeltaAbs"] = (top_metrics["P1 vs. League"] - top_metrics["P2 vs. League"]).abs()
    top_metrics = top_metrics.sort_values("DeltaAbs", ascending=False).head(10)

    gemini_model, gemini_available = initialize_gemini()
    if not gemini_available or gemini_model is None:
        return "AI analýza H2H není dostupná (Gemini není inicializováno)."

    prompt = build_head_to_head_prompt(
        player1_name, player2_name,
        h1, h2,
        sec_delta, sub_delta, top_metrics
    )

    try:
        msg = Content(role="user", parts=[Part.from_text(prompt)])
        config = GenerationConfig(max_output_tokens=4096, temperature=0.5, top_k=30)
        response = gemini_model.generate_content([msg], generation_config=config)
        return response.text
    except Exception as e:
        return f"Došlo k chybě při generování AI H2H analýzy: {e}"