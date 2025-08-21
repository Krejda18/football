# app.py (BigInt-safe AgGrid)
import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import numpy as np
from player_analysis import (
    load_and_process_file, analyze_player, load_all_player_data, 
    calculate_all_player_metrics_and_ratings, run_ai_scout, 
    get_custom_comparison, get_player_comparison_data,
    AVG_DATA_DIR, MIN_MINUTES, COL_POS
)

# --- Cesty ---
DATA_DIR = "./Data"

# --- Hlavní APLIKACE s navigací ---
st.set_page_config(page_title="Skautingový report", page_icon="logo.png", layout="wide")

# --- HLAVIČKA ---
left_col, right_col = st.columns([4, 1])
with right_col:
    st.image("logo.png", width=500)  # Libovolné logo aplikace

st.sidebar.title("Navigace")
app_mode = st.sidebar.radio("Zvolte pohled:", ["Srovnání hráčů", "Detail hráče", "AI Skaut", "Hráč vs. Hráč"])  # přidal jsem i další pohledy pro pohodlí

# --- Zúžení sidebaru ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] { min-width: 200px; max-width: 200px; }
    [data-testid="stSidebar"][aria-expanded="false"] { min-width: 220px; max-width: 220px; margin-left: -220px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Pomocné funkce (BigInt-safe)
# =============================

def aggrid_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Zajistí, že do AgGrid nepůjdou BigInt hodnoty.
    - velká celá čísla → float64
    - extrémně velká čísla (> Number.MAX_SAFE_INTEGER) → string
    """
    if df.empty:
        return df
    df = df.copy()
    # číselné sloupce
    num_cols = df.select_dtypes(include=["number", "int64", "Int64", "uint64"]).columns
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        try:
            max_val = np.nanmax(s.to_numpy(dtype="float64"), initial=np.nan)
        except Exception:
            max_val = np.nan
        if pd.notna(max_val) and max_val > 9_007_199_254_740_991:  # JS Number.MAX_SAFE_INTEGER
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
    st.write(html, unsafe_allow_html=True)


def process_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    processed_df = df.copy()
    first_col_name = processed_df.columns[0]
    mask = processed_df[first_col_name].duplicated()
    processed_df.loc[mask, first_col_name] = ''
    return processed_df

# Styly pro tabulky v detailu hráče
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

    league_files = {file.stem: file for file in sorted(Path(DATA_DIR).glob("*.xlsx"))}
    avg_files = {file.stem: file for file in sorted(Path(AVG_DATA_DIR).glob("*.xlsx"))}
    if not league_files or not avg_files:
        st.error("Chybí datové soubory v adresářích 'data' nebo 'avg_data'.")
        return

    all_avg_dfs = [load_and_process_file(file) for file in avg_files.values()]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]

    col1, col2 = st.columns(2)
    with col1:
        selected_league_name = st.selectbox("Vyber soutěž", options=league_files.keys())

    player_df = load_and_process_file(league_files[selected_league_name])
    player_df_filtered = player_df[player_df["Minutes played"] >= MIN_MINUTES]
    players_list = sorted(player_df_filtered["Player"].dropna().unique())

    with col2:
        selected_player = st.selectbox("Vyber hráče", options=players_list)

    if selected_player and avg_df_filtered is not None:
        result = analyze_player(selected_player, player_df_filtered, avg_df_filtered)

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

        # --- Vlastní srovnání ---
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🆚 Vlastní srovnání</h3>", unsafe_allow_html=True)

        all_positions = sorted(avg_df_filtered[COL_POS].dropna().unique().tolist())
        selected_positions = st.multiselect("Vyberte jednu nebo více pozic pro srovnání:", options=all_positions)

        if selected_positions:
            player_series = player_df_filtered[player_df_filtered['Player'] == selected_player].mean(numeric_only=True)
            main_position = player_df_filtered[player_df_filtered['Player'] == selected_player][COL_POS].iloc[0]
            with st.spinner("Počítám vlastní srovnání..."):
                custom_result = get_custom_comparison(player_series, main_position, selected_positions, avg_df_filtered)

            if "error" in custom_result:
                st.error(custom_result["error"])
            else:
                st.markdown(f"<h4 style='text-align: center;'>Rating vs. Vlastní výběr ({', '.join(selected_positions)})</h4>", unsafe_allow_html=True)
                score_custom = custom_result.get("score", 0)
                st.markdown(
                    f"<div style='font-size:29px; text-align:center; color:{rating_text_color(score_custom)};'><b>{score_custom:.0f} %</b></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<h5 style='text-align: center;'>Sekce</h5>", unsafe_allow_html=True)
                styler_sec_custom = (
                    custom_result["sec_tbl"].style
                    .format("{:.0f}", subset=["vs. Vlastní výběr"])
                    .applymap(background_cells)
                    .set_table_styles(table_style_detail_view)
                    .hide(axis="index")
                )
                render_styled_df(styler_sec_custom)

                st.markdown("<h5 style='text-align: center;'>Podsekce</h5>", unsafe_allow_html=True)
                sub_tbl_custom_processed = process_dataframe_for_display(custom_result["sub_tbl"])
                styler_sub_custom = (
                    sub_tbl_custom_processed.style
                    .format("{:.0f}", subset=["vs. Vlastní výběr"])
                    .applymap(background_cells)
                    .set_table_styles(table_style_detail_view_sub)
                    .hide(axis="index")
                )
                render_styled_df(styler_sub_custom)

        st.markdown("### 🔍 Sekce")
        styler_sec = (
            result["sec_tbl"].style
            .format(format_percent, na_rep="", subset=numeric_cols)
            .applymap(background_cells, subset=numeric_cols)
            .set_table_styles(table_style_detail_view)
            .hide(axis="index")
        )
        render_styled_df(styler_sec)

        st.markdown("### 🛠️ Podsekce")
        sub_tbl_processed = process_dataframe_for_display(result["sub_tbl"])
        styler_sub = (
            sub_tbl_processed.style
            .format(format_percent, na_rep="", subset=numeric_cols)
            .applymap(background_cells, subset=numeric_cols)
            .set_table_styles(table_style_detail_view_sub)
            .hide(axis="index")
        )
        render_styled_df(styler_sub)

        st.markdown("### 📋 Všechny metriky")
        styler_all = (
            result["all_metrics"].style
            .format(format_value, subset=["Hráč", "Liga Ø", "TOP Kluby Ø"], na_rep="")
            .format(format_percent, subset=numeric_cols, na_rep="")
            .applymap(background_cells, subset=numeric_cols)
            .set_table_styles(table_style_detail_view)
            .hide(axis="index")
        )
        render_styled_df(styler_all)

        if st.checkbox("🧠 Zobrazit AI analýzu"):
            if result["gemini_available"]:
                st.markdown(result["analysis"])
            else:
                st.warning(result["analysis"])

# =============================
# Pohled: Srovnání hráčů (AgGrid)
# =============================

def page_player_comparison():
    st.markdown("<h1 style='text-align: center;'>Srovnání hráčů napříč soutěžemi</h1>", unsafe_allow_html=True)

    all_players_df = load_all_player_data()
    avg_files = list(Path(AVG_DATA_DIR).glob("*.xlsx"))
    all_avg_dfs = [load_and_process_file(file) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]

    ratings_df = calculate_all_player_metrics_and_ratings(all_players_df, avg_df_filtered)
    if ratings_df.empty:
        st.warning("Nepodařilo se vypočítat ratingy pro žádné hráče.")
        return

    st.markdown("#### Filtry")
    positions = ["Všechny pozice"] + sorted(ratings_df['Position'].unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_pos = st.selectbox("Filtrovat pozici:", options=positions)
        player_search = st.text_input("Vyhledat hráče:")
    with col2:
        min_age, max_age = int(ratings_df['Age'].min()), int(ratings_df['Age'].max())
        age_range = st.slider("Filtrovat věk:", min_age, max_age, (min_age, max_age))
        min_height, max_height = int(ratings_df['Height'].min()), int(ratings_df['Height'].max())
        height_range = st.slider("Filtrovat výšku (cm):", min_height, max_height, (min_height, max_height))
    with col3:
        min_rating, max_rating = int(ratings_df['Rating vs Liga'].dropna().min()), int(ratings_df['Rating vs Liga'].dropna().max())
        rating_range = st.slider("Filtrovat Rating vs Liga:", min_rating, max_rating, (min_rating, max_rating))

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

    # Aplikace filtrů
    filtered_df = ratings_df.copy()
    if selected_pos != "Všechny pozice":
        filtered_df = filtered_df[filtered_df['Position'] == selected_pos]
    if player_search:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(player_search, case=False, na=False)]

    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1]) &
        (filtered_df['Height'] >= height_range[0]) & (filtered_df['Height'] <= height_range[1]) &
        (filtered_df['Rating vs Liga'] >= rating_range[0]) & (filtered_df['Rating vs Liga'] <= rating_range[1]) &
        (filtered_df['Market value'].fillna(0).between(value_range[0], value_range[1]))
    ]

    st.info(f"Zobrazeno {len(filtered_df)} hráčů. Data v tabulce můžete dále řadit a filtrovat.")

    # ==== JS kódy bezpečné vůči BigInt ====
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

    # CSS zarovnání hlaviček
    custom_css = {
        ".ag-header-cell-label": {"justify-content": "center !important"},
        ".ag-header-cell[col-id='Player'] .ag-header-cell-label": {"justify-content": "flex-start !important"},
        ".ag-header-cell[col-id='Team'] .ag-header-cell-label": {"justify-content": "flex-start !important"},
    }

    # ↓↓↓ BigInt-safe DataFrame pro grid ↓↓↓
    df_for_grid = aggrid_safe_df(filtered_df)

    gb = GridOptionsBuilder.from_dataframe(df_for_grid)
    gb.configure_default_column(sortable=True, filterable=True, resizable=True, wrapHeaderText=True, autoHeaderHeight=True, suppressMenu=True)
    center_aligned_style = {'textAlign': 'center'}

    gb.configure_column("Player", headerName="Hráč", width=200, filter='agSetColumnFilter')
    gb.configure_column("Team", headerName="Tým", width=150, filter='agSetColumnFilter')
    gb.configure_column("League", headerName="Soutěž", cellStyle=center_aligned_style, width=150, filter='agSetColumnFilter')
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

    gridOptions = gb.build()

    AgGrid(
        df_for_grid,
        gridOptions=gridOptions,
        height=900,
        width='100%',
        fit_columns_on_grid_load=False,
        theme='Balham',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        custom_css=custom_css,
    )

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
    avg_files = list(Path(AVG_DATA_DIR).glob("*.xlsx"))
    all_avg_dfs = [load_and_process_file(file) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]

    all_positions = sorted(all_players_df[COL_POS].dropna().unique().tolist())
    selected_pos = st.selectbox("1. Vyberte pozici pro srovnání:", options=all_positions)

    if selected_pos:
        players_on_pos = sorted(all_players_df[all_players_df[COL_POS] == selected_pos]['Player'].unique().tolist())

        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("2. Vyberte prvního hráče:", options=[None] + players_on_pos, index=0)
        with col2:
            player2 = st.selectbox("3. Vyberte druhého hráče:", options=[None] + players_on_pos, index=0)

        if st.button("🔍 Porovnat hráče"):
            if player1 and player2 and player1 != player2:
                with st.spinner(f"Porovnávám hráče {player1} a {player2}..."):
                    comparison_data = get_player_comparison_data(player1, player2, all_players_df, avg_df_filtered)

                res1 = comparison_data["result1"]
                res2 = comparison_data["result2"]
                p1_short = comparison_data["p1_name_short"]
                p2_short = comparison_data["p2_name_short"]

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
                        f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br><span style='color:{rating_text_color(res1['score_lg'])};'><b>{res1['score_lg']:.0f} %</b></span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br><span style='color:{rating_text_color(res1['score_tp'])};'><b>{res1['score_tp']:.0f} %</b></span></div>",
                        unsafe_allow_html=True,
                    )
                with col2_r:
                    st.markdown(
                        f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br><span style='color:{rating_text_color(res2['score_lg'])};'><b>{res2['score_lg']:.0f} %</b></span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br><span style='color:{rating_text_color(res2['score_tp'])};'><b>{res2['score_tp']:.0f} %</b></span></div>",
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
                sec_df = comparison_data["comparison_sec"]
                numeric_cols_sec = sec_df.select_dtypes(include=np.number).columns
                styler_sec = (
                    sec_df.style
                    .format("{:.0f}", subset=numeric_cols_sec, na_rep="–")
                    .apply(style_winner, p1_name=p1_short, p2_name=p2_short, is_ratings=True, axis=None)
                    .set_table_styles(table_style_detail_view)
                    .hide(axis="index")
                )
                render_styled_df(styler_sec)

                st.markdown("<h3 style='text-align: center;'>🆚 Srovnání v podsekcích</h3>", unsafe_allow_html=True)
                sub_df = comparison_data["comparison_sub"]
                sub_processed = process_dataframe_for_display(sub_df)
                numeric_cols_sub = sub_df.select_dtypes(include=np.number).columns
                styler_sub = (
                    sub_processed.style
                    .format("{:.0f}", subset=numeric_cols_sub, na_rep="–")
                    .apply(style_winner, p1_name=p1_short, p2_name=p2_short, is_ratings=True, axis=None)
                    .set_table_styles(table_style_detail_view_sub)
                    .hide(axis="index")
                )
                render_styled_df(styler_sub)

                st.markdown("<h3 style='text-align: center;'>🆚 Srovnání metrik</h3>", unsafe_allow_html=True)
                all_df = comparison_data["comparison_all"]
                numeric_cols_all = all_df.select_dtypes(include=np.number).columns
                styler_all = (
                    all_df.style
                    .format("{:.1f}", subset=numeric_cols_all, na_rep="–")
                    .apply(style_winner, p1_name=p1_short, p2_name=p2_short, is_ratings=False, axis=None)
                    .set_table_styles(table_style_detail_view)
                    .hide(axis="index")
                )
                render_styled_df(styler_all)

            elif not player1 or not player2:
                st.warning("Prosím, vyberte oba hráče pro srovnání.")
            else:
                st.warning("Prosím, vyberte dva různé hráče.")

# =============================
# Router
# =============================

if app_mode == "Detail hráče":
    page_single_player_view()
elif app_mode == "Srovnání hráčů":
    page_player_comparison()
elif app_mode == "AI Skaut":
    page_ai_scout()
elif app_mode == "Hráč vs. Hráč":
    page_player_vs_player()
