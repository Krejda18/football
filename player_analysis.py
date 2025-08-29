# player_analysis.py
import streamlit as st
import pandas as pd
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List
from joblib import Parallel, delayed

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

# --- Paralelizace (konfigurovatelné přes ENV) ---
JOBLIB_N_JOBS = int(os.environ.get("JOBLIB_N_JOBS", "8"))
JOBLIB_BACKEND = os.environ.get("JOBLIB_BACKEND", "loky")  # "loky"=procesy (CPU), "threading"=vlákna (I/O)

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
    files = sorted(Path(DATA_DIR).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    def _load_one(fp: Path) -> pd.DataFrame:
        df_local = load_and_process_file(fp)
        df_local['League'] = fp.stem
        return df_local
    all_player_dfs = Parallel(n_jobs=JOBLIB_N_JOBS, backend=JOBLIB_BACKEND)(delayed(_load_one)(fp) for fp in files)
    if not all_player_dfs:
        return pd.DataFrame()
    combined_df = pd.concat(all_player_dfs, ignore_index=True)
    return combined_df[combined_df["Minutes played"] >= MIN_MINUTES]


def analyze_player(player_name: str, player_df: pd.DataFrame, avg_df: pd.DataFrame, override_position: str | None = None) -> Dict[str, Any]:
    gemini_model, gemini_available = initialize_gemini()
    logic_data = get_logic_definition()
    
    player_rows = player_df[player_df["Player"] == player_name]
    p_avg = player_rows.mean(numeric_only=True)
    detected_position = player_rows[COL_POS].iloc[0]
    main_position = override_position if override_position else detected_position

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

    def _build_metric_row(m: str) -> dict:
        val_lg = (p_avg.get(m, 0) / lg_avg.get(m, 0) * 100) if lg_avg.get(m, 0) != 0 else pd.NA
        val_tp = (p_avg.get(m, 0) / tp_avg.get(m, 0) * 100) if tp_avg.get(m, 0) != 0 else pd.NA
        return {
            "Metric": m,
            "Hráč": p_avg.get(m),
            "Liga Ø": lg_avg.get(m),
            "TOP Kluby Ø": tp_avg.get(m, pd.NA),
            "vs. League": val_lg,
            "vs. TOP 3": val_tp
        }

    rows = Parallel(n_jobs=JOBLIB_N_JOBS, backend=JOBLIB_BACKEND)(delayed(_build_metric_row)(m) for m in sorted(metrics_to_display))
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
    
    def _compute_row(record: dict) -> dict | None:
        main_position_local = record.get(COL_POS)
        if main_position_local not in avg_calcs:
            return None
        p_avg_local = pd.Series(record)
        lg_avg_local = avg_calcs[main_position_local]['lg']
        tp_avg_local = avg_calcs[main_position_local]['tp']
        rat_lg_local = rating_series(p_avg_local, lg_avg_local)
        rat_tp_local = rating_series(p_avg_local, tp_avg_local)
        logic_df_local = logic_flat_df([main_position_local], logic_data)
        logic_metrics_in_data_local = [m for m in logic_df_local["Metric"] if m in rat_lg_local.index]
        score_lg_local = weighted_score(rat_lg_local[logic_metrics_in_data_local], logic_df_local)
        score_tp_local = weighted_score(rat_tp_local[logic_metrics_in_data_local], logic_df_local)
        return {
            'Player': record.get('Player'), 'Team': record.get('Team'),
            'League': record.get('League'), 'Position': main_position_local,
            'Age': record.get('Age'), 'Height': record.get('Height'),
            'Market value': record.get('Market value'),
            'Foot': record.get('Foot'),
            'Minutes': record.get('Minutes played'),
            'Rating vs Liga': score_lg_local, 'Rating vs TOP Kluby': score_tp_local
        }

    records = all_players_df.to_dict('records')
    results = Parallel(n_jobs=JOBLIB_N_JOBS, backend=JOBLIB_BACKEND)(delayed(_compute_row)(rec) for rec in records)
    results = [r for r in results if r is not None]
    
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
    def _enrich_row(record: dict) -> dict | None:
        main_position_local = record.get(COL_POS)
        if main_position_local not in avg_calcs:
            return None
        p_avg_local = pd.Series(record)
        lg_avg_local = avg_calcs[main_position_local]['lg']
        tp_avg_local = avg_calcs[main_position_local]['tp']
        rat_lg_local = rating_series(p_avg_local, lg_avg_local)
        rat_tp_local = rating_series(p_avg_local, tp_avg_local)
        logic_df_local = logic_flat_df([main_position_local], logic_data)
        logic_metrics_in_data_local = [m for m in logic_df_local["Metric"] if m in rat_lg_local.index]
        score_lg_local = weighted_score(rat_lg_local[logic_metrics_in_data_local], logic_df_local)
        player_data_local = dict(record)
        player_data_local['Rating vs Liga'] = score_lg_local
        sec_lg_local, sub_lg_local = breakdown_scores(rat_lg_local, main_position_local, logic_data)
        sec_tp_local, sub_tp_local = breakdown_scores(rat_tp_local, main_position_local, logic_data)
        sec_ratings_local = sec_lg_local.rename(columns={"Score": "lg"}).merge(sec_tp_local.rename(columns={"Score": "tp"}), on="Section", how="outer").to_dict('records')
        sub_ratings_local = sub_lg_local.rename(columns={"Score": "lg"}).merge(sub_tp_local.rename(columns={"Score": "tp"}), on=["Section", "Subsection"], how="outer").to_dict('records')
        player_data_local['sections'] = sec_ratings_local
        player_data_local['subsections'] = sub_ratings_local
        return player_data_local

    records = all_players_df.to_dict('records')
    enriched_results = Parallel(n_jobs=JOBLIB_N_JOBS, backend=JOBLIB_BACKEND)(delayed(_enrich_row)(rec) for rec in records)
    enriched_results = [r for r in enriched_results if r is not None]
    
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
    
    # Průměry pro vybrané pozice (celá liga)
    custom_avg_df = all_avg_df[all_avg_df[COL_POS].isin(custom_positions)]
    if custom_avg_df.empty:
        return {"error": "Pro vybrané pozice nebyla nalezena žádná srovnávací data."}
    custom_avg = custom_avg_df.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)

    # Průměry pro vybrané pozice omezené na TOP kluby
    top_df = custom_avg_df[custom_avg_df["Team"].isin(TOP_CLUBS)]
    if top_df.empty:
        custom_top_avg = pd.Series(dtype='float64')
    else:
        custom_top_avg = top_df.groupby("Player").mean(numeric_only=True).mean(numeric_only=True)

    # Výpočet ratingů
    rat_custom = rating_series(player_series, custom_avg)
    rat_custom_top = rating_series(player_series, custom_top_avg)

    logic_df = logic_flat_df([main_position], logic_data)
    logic_metrics_in_data = [m for m in logic_df["Metric"] if m in rat_custom.index]
    
    score_custom = weighted_score(rat_custom[logic_metrics_in_data], logic_df)
    score_custom_top = weighted_score(rat_custom_top[logic_metrics_in_data], logic_df)

    sec_custom, sub_custom = breakdown_scores(rat_custom, main_position, logic_data)
    sec_custom_top, sub_custom_top = breakdown_scores(rat_custom_top, main_position, logic_data)

    sec_tbl = sec_custom.rename(columns={"Score": "vs. Vlastní výběr"}).merge(
        sec_custom_top.rename(columns={"Score": "vs. Vlastní TOP 3"}), on="Section", how="outer"
    )
    sub_tbl = sub_custom.rename(columns={"Score": "vs. Vlastní výběr"}).merge(
        sub_custom_top.rename(columns={"Score": "vs. Vlastní TOP 3"}), on=["Section", "Subsection"], how="outer"
    )

    return {
        "score_lg": score_custom,
        "score_tp": score_custom_top,
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
