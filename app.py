# app.py (BigInt-safe AgGrid)
import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import numpy as np
from player_analysis import (
    load_and_process_file, analyze_player, load_all_player_data, 
    calculate_all_player_metrics_and_ratings, run_ai_scout, 
    get_custom_comparison, get_player_comparison_data, analyze_head_to_head,
    AVG_DATA_DIR, MIN_MINUTES, COL_POS, DATA_DIR
)
import sys
import os
from st_aggrid import GridUpdateMode

# --- Hlavní APLIKACE s navigací ---
st.set_page_config(page_title="Skautingový report", page_icon="logo.png", layout="wide")

# --- HLAVIČKA ---
left_col, right_col = st.columns([4, 1])
with right_col:
    st.image("logo.png", width=500)

st.sidebar.title("Navigace")
app_mode = st.sidebar.radio("Zvolte pohled:", ["Srovnání hráčů", "Detail hráče", "AI Skaut [Beta]", "Hráč vs. Hráč", "PDF Report"])

# --- Zúžení sidebaru ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] { min-width: 220px; max-width: 220px; }
    [data-testid="stSidebar"][aria-expanded="false"] { min-width: 240px; max-width: 240px; margin-left: -240px; }
    @media print {
        /* Skryj header, sidebar a toolbar ve tisku */
        header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"], .no-print { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Pomocné funkce (BigInt-safe)
# =============================

def aggrid_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    num_cols = df.select_dtypes(include=["number", "int64", "Int64", "uint64"]).columns
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        try:
            max_val = np.nanmax(s.to_numpy(dtype="float64"), initial=np.nan)
        except Exception:
            max_val = np.nan
        if pd.notna(max_val) and max_val > 9_007_199_254_740_991:
            df[c] = df[c].astype("string")
        else:
            df[c] = df[c].astype("float64")
    return df


def rating_text_color(val):
    if pd.isna(val):
        return 'black'
    if val < 95:
        return 'red'
    elif val <= 104:
        return 'orange'
    else:
        return 'green'


def background_cells(val):
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ''
    if val < 95:
        bgcolor = '#ffcccc'
    elif val <= 104:
        bgcolor = '#ffe0b3'
    else:
        bgcolor = '#ccffcc'
    return f'background-color: {bgcolor};'


def render_styled_df(df_styler):
    html = df_styler.to_html(na_rep="")
    # Obleč tabulku do kontejneru, který ji zarovná na střed (bez úvodních mezer, aby se netvořil code-block)
    wrapped = (
        "<div style='display:flex; justify-content:center;'>"
        + "<div style='min-width:60%; max-width:1200px;'>"
        + html
        + "</div>"
        + "</div>"
    )
    st.markdown(wrapped, unsafe_allow_html=True)


def process_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    processed_df = df.copy()
    first_col_name = processed_df.columns[0]
    mask = processed_df[first_col_name].duplicated()
    processed_df.loc[mask, first_col_name] = ''
    return processed_df

table_style_detail_view = [
    {'selector': 'th, td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
    {'selector': 'th', 'props': [('font-weight', 'bold')]},
    {'selector': 'th:first-child, td:first-child', 'props': [('text-align', 'left'), ('font-weight', 'bold')]}
]

table_style_detail_view_sub = [
    {'selector': 'th, td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
    {'selector': 'th', 'props': [('font-weight', 'bold')]},
    {'selector': 'th:nth-child(1), td:nth-child(1)', 'props': [('text-align', 'left'), ('font-weight', 'bold')]},
    {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': [('text-align', 'left')]}
]

# =============================
# Pohled: Detail hráče
# =============================

def page_single_player_view():
    st.markdown("---")

    # <<< ZMĚNA ZDE: Hledáme soubory s koncovkou .parquet >>>
    league_files = {file.stem: file for file in sorted(Path(DATA_DIR).glob("*.parquet"))}
    avg_files = {file.stem: file for file in sorted(Path(AVG_DATA_DIR).glob("*.parquet"))}
    
    if not league_files or not avg_files:
        st.error("Chybí datové soubory v adresářích 'Data_Parquet' nebo 'AVG_Parquet'. Spusťte nejdříve konverzní skript.")
        return

    all_avg_dfs = [load_and_process_file(file) for file in avg_files.values()]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]

    # Ovládací prvky pro výběr soutěže a hráče nejsou součástí tisku
    st.markdown("<div class='no-print'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        selected_league_name = st.selectbox(
            "Vyber soutěž",
            options=[None] + list(league_files.keys()),
            index=0,
            format_func=lambda x: x if x is not None else "— vyberte —",
            key="detail_league_select",
        )

    # Připrav kontext hráčů až po výběru soutěže
    player_df_filtered = None
    players_list = []
    if selected_league_name:
        player_df = load_and_process_file(league_files[selected_league_name])
        player_df_filtered = player_df[player_df["Minutes played"] >= MIN_MINUTES]
        players_list = sorted(player_df_filtered["Player"].dropna().unique())

    with col2:
        if selected_league_name:
            selected_player = st.selectbox(
                "Vyber hráče",
                options=[None] + players_list,
                index=0,
                format_func=lambda x: x if x is not None else "— vyberte —",
                key="detail_player_select",
            )
        else:
            st.info("Nejprve vyberte soutěž.")
            selected_player = None
    st.markdown("</div>", unsafe_allow_html=True)

    # Spusť výpočty až pokud je zvolená soutěž i hráč
    if selected_league_name and selected_player and (player_df_filtered is not None):
        result = analyze_player(selected_player, player_df_filtered, avg_df_filtered)

        # CSS a obal pro tisk – tiskne se pouze tato sekce (od jména hráče), ovládací prvky jsou skryté
        st.markdown(
            """
            <style>
            @media print {
              .no-print, .stButton, .stDownloadButton, [data-testid=stSidebar], [data-testid=stToolbar], header, footer { display: none !important; }
              .block-container { max-width: 100% !important; padding-top: 0 !important; }
              #print-area { position: static !important; width: 100% !important; }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div id='print-area'>", unsafe_allow_html=True)
        st.markdown(result["full_header_block"], unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Vážený rating")

        score_lg_num, score_tp_num = result["score_lg"], result["score_tp"]
        col1_rating, col2_rating = st.columns(2)
        with col1_rating:
            st.markdown(
                f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br><span style='color:{rating_text_color(score_lg_num)};'><b>{score_lg_num:.0f} %</b></span></div>",
                unsafe_allow_html=True,
            )
        with col2_rating:
            st.markdown(
                f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br><span style='color:{rating_text_color(score_tp_num)};'><b>{score_tp_num:.0f} %</b></span></div>",
                unsafe_allow_html=True,
            )

        def format_percent(x):
            if pd.api.types.is_number(x) and not pd.isna(x):
                return f"{x:.0f}"
            return ""

        def format_value(x):
            if pd.api.types.is_number(x) and not pd.isna(x):
                if abs(x) < 10 and x != 0:
                    return f"{x:.2f}"
                if abs(x) < 100:
                    return f"{x:.1f}"
                return f"{x:.0f}"
            return ""

        numeric_cols = ["vs. League", "vs. TOP 3"]

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🆚 Vlastní srovnání</h3>", unsafe_allow_html=True)

        # Nabídni sloučené skupiny pro křídla (AMRL) a krajní beky (FullBack)
        raw_positions = sorted(avg_df_filtered[COL_POS].dropna().unique().tolist())
        base_set = set(raw_positions) - {"AML", "AMR", "DL", "DR"}
        display_options = sorted(base_set.union({"AMRL", "FullBack"}))
        selected_display_positions = st.multiselect("Vyberte jednu nebo více pozic pro srovnání:", options=display_options)

        if selected_display_positions:
            player_series = player_df_filtered[player_df_filtered['Player'] == selected_player].mean(numeric_only=True)
            main_position = player_df_filtered[player_df_filtered['Player'] == selected_player][COL_POS].iloc[0]
            with st.spinner("Počítám vlastní srovnání..."):
                # Rozbal mapované kategorie na konkrétní pozice
                expanded_positions = []
                for pos in selected_display_positions:
                    if pos == "AMRL":
                        expanded_positions.extend(["AML", "AMR"])
                    elif pos == "FullBack":
                        expanded_positions.extend(["DL", "DR"])
                    else:
                        expanded_positions.append(pos)
                # Unikátní pořadí zachováme dle prvního výskytu
                seen = set(); expanded_positions = [p for p in expanded_positions if not (p in seen or seen.add(p))]
                custom_result = get_custom_comparison(player_series, main_position, expanded_positions, avg_df_filtered)

            if "error" in custom_result:
                st.error(custom_result["error"])
            else:
                # Styl pro výraznější oddělení bloku vlastního srovnání
                st.markdown(
                    """
                    <style>
                    .custom-compare-box { border: 2px solid #e6e8eb; border-radius: 10px; padding: 16px; margin: 12px 0 24px 0; background: #fafbfd; }
                    .custom-compare-box h4 { margin-top: 2px; }
                    .custom-compare-divider { border-top: 1px solid #e6e8eb; margin: 12px 0 4px 0; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='custom-compare-box'>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>Rating vs. Vlastní výběr ({', '.join(selected_display_positions)})</h4>", unsafe_allow_html=True)
                score_custom_lg = custom_result.get("score_lg", 0)
                score_custom_tp = custom_result.get("score_tp", 0)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br><span style='color:{rating_text_color(score_custom_lg)};'><b>{score_custom_lg:.0f} %</b></span></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br><span style='color:{rating_text_color(score_custom_tp)};'><b>{score_custom_tp:.0f} %</b></span></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<h5 style='text-align: center;'>Sekce</h5>", unsafe_allow_html=True)
                styler_sec_custom = (
                    custom_result["sec_tbl"].style
                    .format("{:.0f}", subset=["vs. Vlastní výběr", "vs. Vlastní TOP 3"])
                    .applymap(background_cells, subset=["vs. Vlastní výběr", "vs. Vlastní TOP 3"])
                    .set_table_styles(table_style_detail_view)
                    .hide(axis="index")
                )
                render_styled_df(styler_sec_custom)

                st.markdown("<h5 style='text-align: center;'>Podsekce</h5>", unsafe_allow_html=True)
                sub_tbl_custom_processed = process_dataframe_for_display(custom_result["sub_tbl"])
                styler_sub_custom = (
                    sub_tbl_custom_processed.style
                    .format("{:.0f}", subset=["vs. Vlastní výběr", "vs. Vlastní TOP 3"])
                    .applymap(background_cells, subset=["vs. Vlastní výběr", "vs. Vlastní TOP 3"])
                    .set_table_styles(table_style_detail_view_sub)
                    .hide(axis="index")
                )
                render_styled_df(styler_sub_custom)
                st.markdown("<div class='custom-compare-divider'></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center;'>🔍 Sekce</h3>", unsafe_allow_html=True)
        styler_sec = (
            result["sec_tbl"].style
            .format(format_percent, na_rep="", subset=numeric_cols)
            .applymap(background_cells, subset=numeric_cols)
            .set_table_styles(table_style_detail_view)
            .hide(axis="index")
        )
        render_styled_df(styler_sec)

        st.markdown("<h3 style='text-align: center;'>🛠️ Podsekce</h3>", unsafe_allow_html=True)
        sub_tbl_processed = process_dataframe_for_display(result["sub_tbl"])
        styler_sub = (
            sub_tbl_processed.style
            .format(format_percent, na_rep="", subset=numeric_cols)
            .applymap(background_cells, subset=numeric_cols)
            .set_table_styles(table_style_detail_view_sub)
            .hide(axis="index")
        )
        render_styled_df(styler_sub)

        st.markdown("<h3 style='text-align: center;'>📋 Všechny metriky</h3>", unsafe_allow_html=True)
        styler_all = (
            result["all_metrics"].style
            .format(format_value, subset=["Hráč", "Liga Ø", "TOP Kluby Ø"], na_rep="")
            .format(format_percent, subset=numeric_cols, na_rep="")
            .applymap(background_cells, subset=numeric_cols)
            .set_table_styles(table_style_detail_view)
            .hide(axis="index")
        )
        render_styled_df(styler_all)
        # Konec tiskové oblasti – následující ovládací prvky se tisknout nebudou
        st.markdown("</div>", unsafe_allow_html=True)

        # Při změně vybraného hráče zneplatni uloženou AI analýzu, aby se nepřenesla k jinému hráči
        if st.session_state.get("detail_ai_player_name") and st.session_state.get("detail_ai_player_name") != selected_player:
            st.session_state.pop("detail_ai_text", None)
            st.session_state.pop("detail_ai_player_name", None)

        # --- Export PDF ---
        st.markdown("---")
        st.markdown("### 📄 Export do PDF")
        col_export_no_ai, col_export_ai = st.columns(2)
        with col_export_no_ai:
            export_no_ai_clicked = st.button("Exportovat PDF (bez AI)")
        with col_export_ai:
            ai_ready = (
                "detail_ai_text" in st.session_state and bool(st.session_state["detail_ai_text"]) and
                st.session_state.get("detail_ai_player_name") == selected_player
            )
            export_with_ai_clicked = st.button("Exportovat PDF (s AI)", type="primary", disabled=not ai_ready)
            if not ai_ready:
                st.caption("Nejprve vygenerujte AI analýzu pro právě vybraného hráče.")

        def _do_export(gemini_texts: dict):
            from pdf_export import (
                extract_and_process_data_for_pdf,
                generate_player_report_pdf_full,
                _load_combined_avg_dataframe,
            )
            with st.spinner("Generuji PDF report…"):
                df_combined_avg = _load_combined_avg_dataframe()
                if df_combined_avg is None or df_combined_avg.empty:
                    st.error("AVG průměry nejsou dostupné. Zkontrolujte složku AVG_Parquet.")
                    return
                processed = extract_and_process_data_for_pdf(selected_player, player_df_filtered, df_combined_avg)
                if not processed:
                    st.error("Nelze připravit data pro PDF pro vybraného hráče.")
                    return
                import tempfile, os
                import datetime as _dt
                safe_name = processed.get("player_name", selected_player).replace(" ", "_").replace(".", "")
                filename = f"{safe_name}_{_dt.date.today().strftime('%Y-%m-%d')}_performance_report.pdf"
                with tempfile.TemporaryDirectory() as _tmpdir:
                    out_path = os.path.join(_tmpdir, filename)
                    generate_player_report_pdf_full(processed, gemini_texts, out_path)
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            st.download_button(
                                label="Stáhnout PDF",
                                data=f.read(),
                                file_name=filename,
                                mime="application/pdf",
                            )
                    else:
                        st.error("PDF soubor se nepodařilo vytvořit.")

        if export_no_ai_clicked:
            _do_export({"cs": "Textová analýza Gemini AI není k dispozici.", "en": ""})
        if export_with_ai_clicked and ai_ready:
            _do_export({"cs": st.session_state["detail_ai_text"], "en": ""})

        # Pokud již existuje vygenerovaná AI analýza pro aktuálně vybraného hráče, vypiš ji níže
        if (
            st.session_state.get("detail_ai_text") and
            st.session_state.get("detail_ai_player_name") == selected_player
        ):
            st.markdown("---")
            st.markdown(st.session_state["detail_ai_text"])

        if st.button("🧠 Vygenerovat AI analýzu", type="primary"):
            with st.spinner("Generuji AI analýzu..."):
                from player_analysis import generate_ai_analysis
                ai_text = generate_ai_analysis(selected_player, result["sec_tbl"], result["sub_tbl"], result["all_metrics"], [result["main_position"]])
            st.session_state["detail_ai_text"] = ai_text
            st.session_state["detail_ai_player_name"] = selected_player
            st.success("AI analýza vygenerována. Můžete exportovat PDF s analýzou.")
            st.markdown(ai_text)
            # Okamžitě znovu vykresli stránku, aby se exportní tlačítko odemklo
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

# =============================
# Pohled: Srovnání hráčů (AgGrid)
# =============================

def page_player_comparison():
    st.markdown("<h1 style='text-align: center;'>Srovnání hráčů napříč soutěžemi</h1>", unsafe_allow_html=True)

    all_players_df = load_all_player_data()
    avg_files = list(Path(AVG_DATA_DIR).glob("*.parquet"))
    
    # Přidej sloupec League podle názvu souboru, aby bylo možné filtrovat sezóny
    all_avg_dfs = [load_and_process_file(file).assign(League=file.stem) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]

    ratings_df = calculate_all_player_metrics_and_ratings(all_players_df, avg_df_filtered)
    if ratings_df.empty:
        st.warning("Nepodařilo se vypočítat ratingy pro žádné hráče.")
        return

    st.markdown("#### Filtry")
    positions = ["Všechny pozice"] + sorted(ratings_df['Position'].unique().tolist())
    leagues_all = sorted(ratings_df['League'].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_pos = st.selectbox("Filtrovat pozici:", options=positions)
        player_search = st.text_input("Vyhledat hráče:")
    with col2:
        min_age, max_age = int(ratings_df['Age'].min()), int(ratings_df['Age'].max())
        # Výchozí věkový filtr: pod 22 let
        default_age_max = min(21, max_age)
        age_range = st.slider("Filtrovat věk:", min_age, max_age, (min_age, default_age_max))
        min_height, max_height = int(ratings_df['Height'].min()), int(ratings_df['Height'].max())
        height_range = st.slider("Filtrovat výšku (cm):", min_height, max_height, (min_height, max_height))
    with col3:
        # Bezpečný rozsah pro slider ratingu (očekáváme ~ 50–150, ale ošetříme extrémy/typy)
        _rat = pd.to_numeric(ratings_df['Rating vs Liga'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if _rat.empty:
            min_rating, max_rating = 50, 150
        else:
            min_raw, max_raw = float(_rat.min()), float(_rat.max())
            # Svážeme do smysluplného rozsahu a zaokrouhlíme
            min_rating = int(max(0, np.floor(min(min_raw, 50))))
            max_rating = int(min(200, np.ceil(max(max_raw, 150))))
        rating_range = st.slider("Filtrovat Rating vs Liga:", min_rating, max_rating, (min_rating, max_rating), step=1)
        # Kompaktní výběr soutěží v popoveru, aby se nezahltil layout
        ss = st.session_state
        ss.setdefault("cmp_league_filter", leagues_all)
        # Odvoď ročníky z nového názvu souboru: soutez_sezona_datum
        # Bereme prostřední token (sezona), např. "Czechia_24-25_20250101" -> "24-25"
        def season_of(name: str) -> str:
            if not isinstance(name, str):
                return "ostatní"
            parts = name.split("_")
            if len(parts) >= 2:
                token = parts[1]
            else:
                token = name
            return token.replace("_", "-").replace("/", "-")
        seasons_all = sorted({season_of(l) for l in leagues_all})
        default_seasons = [s for s in ["25-26", "25"] if s in seasons_all]
        ss.setdefault("cmp_league_seasons", default_seasons if default_seasons else seasons_all)
        pop = st.popover("Soutěž", use_container_width=True)
        with pop:
            st.markdown("**Ročníky:**")
            cols_season = st.columns([1,1,3])
            with cols_season[0]:
                if st.button("Všechny ročníky"):
                    ss["cmp_league_seasons"] = seasons_all
            with cols_season[1]:
                if st.button("Vyčistit ročníky"):
                    ss["cmp_league_seasons"] = []
            ss["cmp_league_seasons"] = st.multiselect(
                "",
                options=seasons_all,
                default=[s for s in ss["cmp_league_seasons"] if s in seasons_all],
                key="cmp_league_seasons_ms"
            )
            search_lg = st.text_input("Hledat soutěž:", placeholder="např. Czechia nebo 24-25")
            if st.button("Vybrat vše"):
                ss["cmp_league_filter"] = leagues_all
            if st.button("Vyčistit"):
                ss["cmp_league_filter"] = []
            # Omez nabídku dle ročníku a vyhledávání
            if ss.get("cmp_league_seasons"):
                base = [l for l in leagues_all if season_of(l) in ss["cmp_league_seasons"]]
            else:
                base = leagues_all
            if search_lg:
                opt = [l for l in base if search_lg.lower() in l.lower()]
            else:
                opt = base
            ss["cmp_league_filter"] = st.multiselect(
                "Vyber soutěže:",
                options=opt,
                default=[l for l in ss["cmp_league_filter"] if l in opt],
                key="cmp_league_filter_ms"
            )
        st.caption(f"Ročníky: {len(ss.get('cmp_league_seasons', []))}/{len(seasons_all)} • Soutěže: {len(ss['cmp_league_filter'])}/{len(leagues_all)}")

    st.markdown("---")
    max_val = int(ratings_df['Market value'].fillna(0).max())
    steps1 = np.arange(0, 1_000_001, 25_000)
    steps2 = np.arange(1_000_000, 5_000_001, 500_000)
    steps3 = np.arange(5_000_000, max_val + 1_000_000, 1_000_000)
    market_value_steps = np.unique(np.concatenate((steps1, steps2, steps3))).tolist()

    def format_market_value(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} M €"
        return f"{value / 1000:.0f} tis. €"

    value_range = st.select_slider(
        "Filtrovat tržní hodnotu:",
        options=market_value_steps,
        value=(market_value_steps[0], market_value_steps[-1]),
        format_func=format_market_value,
    )

    filtered_df = ratings_df.copy()
    if selected_pos != "Všechny pozice":
        filtered_df = filtered_df[filtered_df['Position'] == selected_pos]
    if player_search:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(player_search, case=False, na=False)]
    if 'cmp_league_filter' in st.session_state and st.session_state['cmp_league_filter']:
        filtered_df = filtered_df[filtered_df['League'].isin(st.session_state['cmp_league_filter'])]
    
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1]) &
        (filtered_df['Height'] >= height_range[0]) & (filtered_df['Height'] <= height_range[1]) &
        (filtered_df['Rating vs Liga'] >= rating_range[0]) & (filtered_df['Rating vs Liga'] <= rating_range[1]) &
        (filtered_df['Market value'].fillna(0).between(value_range[0], value_range[1]))
    ]

    st.info(f"Zobrazeno {len(filtered_df)} hráčů. Data v tabulce můžete dále řadit a filtrovat.")

    safeNumberFormatter = JsCode(
        """
        function(params){
          const v = Number(params.value);
          if (!isFinite(v)) return params.value ?? '';
          return v.toFixed(0);
        }
        """
    )

    percentFormatter = JsCode(
        """
        function(params){
          const v = Number(params.value);
          if (!isFinite(v)) return '';
          return v.toFixed(0) + ' %';
        }
        """
    )

    bar_chart_renderer = JsCode(
        """
        class BarChartCellRenderer {
            init(params) {
                this.eGui = document.createElement('div');
                this.eGui.style.width = '100%';
                this.eGui.style.height = '100%';
                this.eGui.style.position = 'relative';
                this.eGui.style.backgroundColor = '#f0f2f6';

                this.bar = document.createElement('div');
                this.bar.style.height = '100%';

                this.label = document.createElement('div');
                this.label.style.position = 'absolute';
                this.label.style.top = '50%';
                this.label.style.left = '50%';
                this.label.style.transform = 'translate(-50%, -50%)';
                this.label.style.fontWeight = 'bold';
                this.label.style.color = '#333';

                this.eGui.appendChild(this.bar);
                this.eGui.appendChild(this.label);
                this.refresh(params);
            }
            getGui() { return this.eGui; }
            refresh(params) {
                const raw = params.value;
                const value = Number(raw);
                if (!isFinite(value)) { this.label.innerHTML = ''; this.bar.style.width = '0%'; return true; }

                let color;
                if (value < 90) { color = '#ffcdd2'; }
                else if (value < 100) { color = '#fff9c4'; }
                else { color = '#c8e6c9'; }

                this.bar.style.backgroundColor = color;
                this.bar.style.width = Math.min(100, (value / 150 * 100)) + '%';
                this.label.innerHTML = Math.round(value);
                return true;
            }
        }
        """
    )

    market_value_formatter = JsCode(
        """
        function(params) {
            const v = Number(params.value);
            if (!isFinite(v)) return '';
            try {
                return v.toLocaleString('cs-CZ', {style: 'currency', currency: 'EUR', maximumFractionDigits: 0});
            } catch(e) {
                return v;
            }
        }
        """
    )

    custom_css = {
        ".ag-header-cell-label": {"justify-content": "center !important"},
        ".ag-header-cell[col-id='Player'] .ag-header-cell-label": {"justify-content": "flex-start !important"},
        ".ag-header-cell[col-id='Team'] .ag-header-cell-label": {"justify-content": "flex-start !important"},
    }

    df_for_grid = aggrid_safe_df(filtered_df)

    gb = GridOptionsBuilder.from_dataframe(df_for_grid)
    gb.configure_default_column(sortable=True, filterable=True, resizable=True, wrapHeaderText=True, autoHeaderHeight=True, suppressMenu=True)
    center_aligned_style = {'textAlign': 'center'}

    gb.configure_column("Player", headerName="Hráč", width=200, filter='agSetColumnFilter')
    gb.configure_column("Team", headerName="Tým", width=150, filter='agSetColumnFilter')
    left_aligned_style = {'textAlign': 'left'}
    gb.configure_column("League", headerName="Soutěž", cellStyle=left_aligned_style, width=180, filter='agSetColumnFilter')
    gb.configure_column("Position", headerName="Pozice", cellStyle=center_aligned_style, width=100, filter=False)
    gb.configure_column("Age", headerName="Věk", cellStyle=center_aligned_style, width=80, filter=False, valueFormatter=safeNumberFormatter)
    gb.configure_column("Height", headerName="Výška", cellStyle=center_aligned_style, width=80, filter=False, valueFormatter=safeNumberFormatter)
    gb.configure_column("Minutes", headerName="Minuty", cellStyle=center_aligned_style, width=90, filter=False, valueFormatter=safeNumberFormatter)

    gb.configure_column(
        "Market value",
        headerName="Tržní hodnota",
        valueFormatter=market_value_formatter,
        cellStyle=center_aligned_style,
        width=150,
        filter=False,
    )

    gb.configure_column("Rating vs Liga", cellRenderer=bar_chart_renderer, width=150, filter=False)
    gb.configure_column("Rating vs TOP Kluby", cellRenderer=bar_chart_renderer, width=180, filter=False)

    # Obnova uloženého stavu filtrů/sortů, pokud existuje
    ss = st.session_state
    saved_state = ss.get("cmp_grid_state")
    if saved_state and isinstance(saved_state, dict):
        try:
            gb.configure_grid_options(columnState=saved_state.get("columns_state"))
        except Exception:
            pass
    else:
        # Výchozí řazení: nejlepší nahoře podle "Rating vs Liga"
        try:
            gb.configure_column("Rating vs Liga", sort='desc', sortIndex=0)
        except Exception:
            pass
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_for_grid,
        gridOptions=gridOptions,
        height=900,
        width='100%',
        fit_columns_on_grid_load=False,
        theme='Balham',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        custom_css=custom_css,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        key="comparison_aggrid",
    )

    # Ulož stav sloupců/filtrů, pokud ho komponenta vrátí
    if isinstance(grid_response, dict) and "grid_state" in grid_response:
        ss["cmp_grid_state"] = grid_response["grid_state"]

# =============================
# Pohled: AI Skaut
# =============================

def page_ai_scout():
    st.header("🤖 AI Skaut")
    st.info("Zadejte do textového pole níže vaše požadavky na hráče a AI prohledá databázi a navrhne nejlepší kandidáty.")

    default_prompt = (
        "Hledám mladého (pod 23 let) ofenzivního záložníka (AMC) s vysokým ratingem proti TOP klubům. "
        "Měl by být kreativní a mít potenciál pro další růst."
    )

    user_needs = st.text_area("Popište ideálního hráče:", height=150, value=default_prompt)

    if st.button("🔍 Najít hráče"):
        with st.spinner("AI analyzuje data a hledá nejlepší shody..."):
            recommendation = run_ai_scout(user_needs)
            st.markdown("---")
            st.subheader("Doporučení od AI Skauta:")
            st.markdown(recommendation)

# =============================
# Pohled: Hráč vs. Hráč
# =============================

def page_player_vs_player():
    st.markdown("---")
    st.header("👥 Hráč vs. Hráč")

    all_players_df = load_all_player_data()
    avg_files = list(Path(AVG_DATA_DIR).glob("*.parquet"))
    all_avg_dfs = [load_and_process_file(file) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]

    # Helper: nový formát "soutez_sezona_datum" -> vrať prostřední token jako sezónu
    def extract_season_token(league_name: str) -> str | None:
        if not isinstance(league_name, str):
            return None
        parts = league_name.split("_")
        if len(parts) >= 2:
            return parts[1]
        return None

    all_positions = sorted(all_players_df[COL_POS].dropna().unique().tolist())
    selected_pos = st.selectbox("1. Vyberte pozici pro srovnání:", options=all_positions)
    if not selected_pos:
        return

    players_on_pos = sorted(all_players_df[all_players_df[COL_POS] == selected_pos]['Player'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("2. Vyberte prvního hráče:", options=[None] + players_on_pos, index=0, key="h2h_p1")
        if player1:
            leagues_p1 = all_players_df.loc[all_players_df['Player'] == player1, 'League'].dropna().unique().tolist()
            seasons_p1 = sorted({extract_season_token(l) or l for l in leagues_p1})
            season1 = st.selectbox(
                "Sezóna hráče 1:",
                options=[None] + seasons_p1,
                index=0,
                format_func=lambda x: x if x is not None else "— všechny —",
                key="h2h_season_p1",
            )
        else:
            season1 = None
    with col2:
        player2 = st.selectbox("3. Vyberte druhého hráče:", options=[None] + players_on_pos, index=0, key="h2h_p2")
        if player2:
            leagues_p2 = all_players_df.loc[all_players_df['Player'] == player2, 'League'].dropna().unique().tolist()
            seasons_p2 = sorted({extract_season_token(l) or l for l in leagues_p2})
            season2 = st.selectbox(
                "Sezóna hráče 2:",
                options=[None] + seasons_p2,
                index=0,
                format_func=lambda x: x if x is not None else "— všechny —",
                key="h2h_season_p2",
            )
        else:
            season2 = None

    ss = st.session_state
    ss.setdefault("h2h_compared", False)
    ss.setdefault("h2h_pair", (None, None))
    ss.setdefault("h2h_seasons", (None, None))
    ss.setdefault("h2h_last_comp", None)
    ss.setdefault("h2h_ai_text", None)

    if st.button("🔍 Porovnat hráče", type="primary"):
        if not player1 or not player2:
            st.warning("Prosím, vyberte oba hráče pro srovnání.")
        elif player1 == player2 and season1 == season2:
            st.warning("Prosím, vyberte buď dva různé hráče, nebo stejného hráče v různých sezónách.")
        else:
            df_h2h = all_players_df.copy()

            # Pokud je vybraná sezóna, filtruj řádky daného hráče na soutěže obsahující token sezóny
            def filter_by_season(df: pd.DataFrame, player: str, season_sel: str | None) -> pd.DataFrame:
                if not season_sel:
                    return df
                token = season_sel
                return df[~((df['Player'] == player) & (~df['League'].astype(str).str.contains(token, na=False)))]

            df_h2h = filter_by_season(df_h2h, player1, extract_season_token(season1) if season1 else None)
            df_h2h = filter_by_season(df_h2h, player2, extract_season_token(season2) if season2 else None)

            with st.spinner(f"Porovnávám hráče {player1} a {player2}..."):
                comp = get_player_comparison_data(
                    player1,
                    player2,
                    df_h2h,
                    avg_df_filtered,
                    season1 if season1 else None,
                    season2 if season2 else None,
                )

            ss.h2h_compared = True
            # Ulož informaci o hráčích a sezónách pro správné rozpoznání změn
            ss.h2h_pair = (player1, player2)
            ss.h2h_seasons = (season1, season2)
            ss.h2h_last_comp = comp
            ss.h2h_ai_text = None

    if ss.h2h_compared and ss.h2h_last_comp is not None:
        comp = ss.h2h_last_comp
        res1, res2 = comp["result1"], comp["result2"]
        p1_short, p2_short = comp["p1_name_short"], comp["p2_name_short"]

        st.markdown("---")
        col1_h, col2_h = st.columns(2)
        with col1_h:
            st.markdown(res1["full_header_block"], unsafe_allow_html=True)
        with col2_h:
            st.markdown(res2["full_header_block"], unsafe_allow_html=True)

        st.markdown("### 📈 Vážený rating")
        col1_r, col2_r = st.columns(2)
        with col1_r:
            st.markdown(
                f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br>"
                f"<span style='color:{rating_text_color(res1['score_lg'])};'><b>{res1['score_lg']:.0f} %</b></span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br>"
                f"<span style='color:{rating_text_color(res1['score_tp'])};'><b>{res1['score_tp']:.0f} %</b></span></div>",
                unsafe_allow_html=True,
            )
        with col2_r:
            st.markdown(
                f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br>"
                f"<span style='color:{rating_text_color(res2['score_lg'])};'><b>{res2['score_lg']:.0f} %</b></span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br>"
                f"<span style='color:{rating_text_color(res2['score_tp'])};'><b>{res2['score_tp']:.0f} %</b></span></div>",
                unsafe_allow_html=True,
            )

        def style_winner(df, p1_name, p2_name, is_ratings=True):
            styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
            highlight = 'background-color: lightgreen'
            p1_cols = [c for c in df.columns if p1_name in c]
            p2_cols = [c for c in df.columns if p2_name in c]
            if not (p1_cols and p2_cols):
                return styled_df
            if is_ratings:
                for p1_col, p2_col in zip(p1_cols, p2_cols):
                    numeric_p1 = pd.to_numeric(df[p1_col], errors='coerce')
                    numeric_p2 = pd.to_numeric(df[p2_col], errors='coerce')
                    styled_df[p1_col] = np.where(numeric_p1 > numeric_p2, highlight, '')
                    styled_df[p2_col] = np.where(numeric_p2 > numeric_p1, highlight, '')
            else:
                p1_col, p2_col = p1_cols[0], p2_cols[0]
                numeric_p1 = pd.to_numeric(df[p1_col], errors='coerce')
                numeric_p2 = pd.to_numeric(df[p2_col], errors='coerce')
                styled_df[p1_col] = np.where(numeric_p1 > numeric_p2, highlight, '')
                styled_df[p2_col] = np.where(numeric_p2 > numeric_p1, highlight, '')
            return styled_df

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🆚 Srovnání v sekcích</h3>", unsafe_allow_html=True)
        sec_df = comp["comparison_sec"]
        numeric_cols_sec = sec_df.select_dtypes(include=np.number).columns
        styler_sec = (
            sec_df.style
            .format("{:.0f}", subset=numeric_cols_sec, na_rep="–")
            .applymap(background_cells, subset=numeric_cols_sec)
            .apply(style_winner, p1_name=p1_short, p2_name=p2_short, is_ratings=True, axis=None)
            .set_table_styles(table_style_detail_view)
            .hide(axis="index")
        )
        render_styled_df(styler_sec)

        st.markdown("<h3 style='text-align: center;'>🆚 Srovnání v podsekcích</h3>", unsafe_allow_html=True)
        sub_df = comp["comparison_sub"]
        sub_processed = process_dataframe_for_display(sub_df)
        numeric_cols_sub = sub_df.select_dtypes(include=np.number).columns
        styler_sub = (
            sub_processed.style
            .format("{:.0f}", subset=numeric_cols_sub, na_rep="–")
            .applymap(background_cells, subset=numeric_cols_sub)
            .apply(style_winner, p1_name=p1_short, p2_name=p2_short, is_ratings=True, axis=None)
            .set_table_styles(table_style_detail_view_sub)
            .hide(axis="index")
        )
        render_styled_df(styler_sub)

        st.markdown("<h3 style='text-align: center;'>🆚 Srovnání metrik</h3>", unsafe_allow_html=True)
        all_df = comp["comparison_all"].copy()
        metric_num_cols = [c for c in all_df.columns if c in (p1_short, p2_short)]
        styler_all = (
            all_df.style
            .format("{:.1f}", subset=metric_num_cols, na_rep="–")
            .applymap(background_cells, subset=metric_num_cols)
            .apply(style_winner, p1_name=p1_short, p2_name=p2_short, is_ratings=False, axis=None)
            .set_table_styles(table_style_detail_view)
            .hide(axis="index")
        )
        render_styled_df(styler_all)

        st.markdown("---")
        st.subheader("🧠 AI H2H analýza")

        # Zkontroluj, zda se změnili hráči nebo sezóny
        current_pair = (player1, player2)
        current_seasons = (season1, season2)
        same_comparison = (ss.h2h_pair == current_pair and 
                          ss.get("h2h_seasons", (None, None)) == current_seasons)
        
        if same_comparison:
            if st.button("Vygenerovat AI porovnání"):
                with st.spinner("AI tvoří porovnání..."):
                    ss.h2h_ai_text = analyze_head_to_head(player1, player2, all_players_df, avg_df_filtered)

            if ss.h2h_ai_text:
                st.markdown(ss.h2h_ai_text, unsafe_allow_html=True)
        else:
            st.info("Změnili jste výběr hráčů. Nejdřív klikněte na „Porovnat hráče“, pak spusťte AI porovnání.")
            

# =============================
# PDF Report
# =============================

def page_pdf_report():
    """Stránka pro generování PDF reportů z XLSX souboru"""
    st.title("📄 Generování PDF Reportů")
    
    st.markdown("""
    ### Jak to funguje:
    1. **Nahrajte XLSX soubor** s daty hráčů
    2. **Klikněte na "Generovat PDF"** 
    3. **Stáhněte si** vygenerované PDF reporty přímo z prohlížeče
    
    **📝 Důležité:** Pro každého hráče z nahraného souboru se vygeneruje samostatný PDF report.
    
    PDF reporty budou obsahovat:
    - 📊 Celkový rating vs. Liga a TOP kluby
    - 📈 Detailní analýzu podle sekcí a sub-sekcí
    - 🤖 AI analýzu výkonnosti hráče
    - 📋 Přehled všech relevantních metrik
    """)
    
    # Nahrání souboru
    uploaded_file = st.file_uploader(
        "Vyberte XLSX soubor s daty hráčů:",
        type=['xlsx'],
        help="Soubor musí obsahovat sloupce: Player, Team, Position, Minutes played a další metriky"
    )
    
    if uploaded_file is not None:
        try:
            # Uložení dočasného souboru
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Soubor '{uploaded_file.name}' byl úspěšně nahrán!")
            # Načtení jmen hráčů ze souboru pro výběr konkrétního reportu
            try:
                df_preview = pd.read_excel(temp_file_path, engine="openpyxl")
                player_options = sorted([p for p in df_preview.get("Player", pd.Series(dtype=str)).dropna().astype(str).unique()])
            except Exception:
                player_options = []

            selected_player = None
            if player_options:
                selected_player = st.selectbox("Vyberte hráče pro PDF report (volitelné):", ["(první v souboru)"] + player_options)
                if selected_player == "(první v souboru)":
                    selected_player = None
            
            # Tlačítko pro generování PDF
            if st.button("🚀 Generovat PDF Reporty", type="primary"):
                with st.spinner("Generuji PDF reporty... Může to trvat několik minut."):
                    try:
                        # Import PDF exportu (detailní varianta vrací i důvody přeskočení)
                        from pdf_export import generate_all_players_reports_web_detailed
                        
                        # Vytvoření dočasné složky pro PDF soubory
                        temp_pdf_dir = "temp_pdf_reports"
                        os.makedirs(temp_pdf_dir, exist_ok=True)
                        
                        # Spuštění generování s dočasnou složkou (detailní výstup)
                        generated_files, skipped = generate_all_players_reports_web_detailed(temp_file_path, temp_pdf_dir)
                        
                        if generated_files:
                            st.success(f"🎉 PDF reporty byly úspěšně vygenerovány! ({len(generated_files)} souborů)")
                            
                            # Zobrazení informace o počtu hráčů
                            st.info(f"📊 Bylo zpracováno {len(generated_files)} hráčů z nahraného souboru.")
                            
                            st.markdown("### 📥 Stáhnout PDF Reporty:")
                            
                            for pdf_file in generated_files:
                                # Získání názvu souboru
                                filename = os.path.basename(pdf_file)
                                
                                # Načtení PDF souboru
                                with open(pdf_file, "rb") as f:
                                    pdf_data = f.read()
                                
                                # Tlačítko pro stažení
                                st.download_button(
                                    label=f"📄 Stáhnout {filename}",
                                    data=pdf_data,
                                    file_name=filename,
                                    mime="application/pdf",
                                    key=f"download_{filename}"
                                )
                        # Zobrazení informací o přeskočených hráčích (pokud existují)
                        if 'skipped' in locals() and skipped:
                            st.markdown("### ⚠️ Přeskočení hráči a důvody")
                            try:
                                st.dataframe(pd.DataFrame(skipped), use_container_width=True)
                            except Exception:
                                for item in skipped:
                                    st.write(f"- {item.get('player', 'N/A')}: {item.get('reason', 'bez důvodu')}")
                        if not generated_files:
                            st.warning("⚠️ Nebyly vygenerovány žádné PDF soubory.")
                        
                    except Exception as e:
                        st.error(f"❌ Chyba při generování PDF: {str(e)}")
                        st.exception(e)
                    finally:
                        # Smazání dočasného souboru
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            
        except Exception as e:
            st.error(f"❌ Chyba při nahrávání souboru: {str(e)}")
            st.exception(e)
    
    # Informace o formátu souboru
    with st.expander("📋 Požadovaný formát XLSX souboru"):
        st.markdown("""
        **Povinné sloupce:**
        - `Player` - Jméno hráče
        - `Team` - Název klubu  
        - `Position` - Pozice hráče
        - `Minutes played` - Odehrané minuty
        - `Age` - Věk hráče
        - `Height` - Výška (cm)
        - `Weight` - Váha (kg)
        
        **Doporučené metriky:**
        - Všechny metriky s příponou "per 90"
        - Procentuální metriky (úspěšnost, přesnost)
        - Defenzivní a ofenzivní metriky
        - Metriky pro přihrávky a střelbu
        
        **Poznámka:** Hráči s méně než 300 odehranými minutami budou automaticky vyřazeni z analýzy.
        """)

# =============================
# Router
# =============================

if app_mode == "Detail hráče":
    page_single_player_view()
elif app_mode == "Srovnání hráčů":
    page_player_comparison()
elif app_mode in ("AI Skaut", "AI Skaut [Beta]"):
    page_ai_scout()
elif app_mode == "Hráč vs. Hráč":
    page_player_vs_player()
elif app_mode == "PDF Report":
    page_pdf_report()
