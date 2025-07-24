# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import numpy as np
from player_analysis import (
    load_and_process_file, analyze_player, load_all_player_data, calculate_ratings_for_all_players,
    AVG_DATA_DIR, MIN_MINUTES
)

# --- Cesty ---
DATA_DIR = "./Data"

# --- Pomocné funkce a styly ---
def rating_text_color(val):
    if pd.isna(val): return 'black'
    if val < 95: return 'red'
    elif val <= 104: return 'orange'
    else: return 'green'

def background_cells(val):
    if pd.isna(val) or not isinstance(val, (int, float)): return ''
    if val < 95: bgcolor = '#ffcccc'
    elif val <= 104: bgcolor = '#ffe0b3'
    else: bgcolor = '#ccffcc'
    return f'background-color: {bgcolor};'

def render_styled_df(df_styler):
    html = df_styler.to_html(na_rep="")
    st.write(html, unsafe_allow_html=True)

def process_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    processed_df = df.copy()
    first_col_name = processed_df.columns[0]
    mask = processed_df[first_col_name].duplicated()
    processed_df.loc[mask, first_col_name] = ''
    return processed_df

# Styly pro tabulky v detailu hráče
table_style_detail_view = [{'selector': 'th, td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]}, {'selector': 'th', 'props': [('font-weight', 'bold')]}, {'selector': 'th:first-child, td:first-child', 'props': [('text-align', 'left'), ('font-weight', 'bold')]}]
table_style_detail_view_sub = [{'selector': 'th, td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]}, {'selector': 'th', 'props': [('font-weight', 'bold')]}, {'selector': 'th:nth-child(1), td:nth-child(1)', 'props': [('text-align', 'left'), ('font-weight', 'bold')]}, {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': [('text-align', 'left')]}]

# --- Funkce pro jednotlivé stránky ---
def page_single_player_view():
    st.markdown("<h1 style='text-align: center;'>📊 Skautingový report hráče</h1>", unsafe_allow_html=True)
    league_files = {file.stem: file for file in sorted(Path(DATA_DIR).glob("*.xlsx"))}
    avg_files = {file.stem: file for file in sorted(Path(AVG_DATA_DIR).glob("*.xlsx"))}
    if not league_files or not avg_files:
        st.error("Chybí datové soubory v adresářích 'Data' nebo 'AVG - hodnoty'.")
        return
    all_avg_dfs = [load_and_process_file(file) for file in avg_files.values()]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]
    
    # --- ZMĚNA ZDE: Rozložení výběrových menu ---
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            selected_league_name = st.selectbox("Vyber soutěž", options=league_files.keys())
        
        player_df = load_and_process_file(league_files[selected_league_name])
        player_df_filtered = player_df[player_df["Minutes played"] >= MIN_MINUTES]
        players_list = sorted(player_df_filtered["Player"].dropna().unique())
        
        with sub_col2:
            selected_player = st.selectbox("Vyber hráče", options=players_list)

    if selected_player and avg_df_filtered is not None:
        result = analyze_player(selected_player, player_df_filtered, avg_df_filtered)
        
        st.markdown(result["full_header_block"], unsafe_allow_html=True)

        # Zbytek stránky pokračuje beze změny
        st.markdown("---")
        st.markdown("### 📈 Vážený rating")
        
        score_lg_num, score_tp_num = result["score_lg"], result["score_tp"]
        col1_rating, col2_rating = st.columns(2)
        with col1_rating: st.markdown(f"<div style='font-size:29px; text-align:center;'>Vs. Liga<br><span style='color:{rating_text_color(score_lg_num)};'><b>{score_lg_num:.0f} %</b></span></div>", unsafe_allow_html=True)
        with col2_rating: st.markdown(f"<div style='font-size:29px; text-align:center;'>Vs. TOP Kluby<br><span style='color:{rating_text_color(score_tp_num)};'><b>{score_tp_num:.0f} %</b></span></div>", unsafe_allow_html=True)
        def format_percent(x):
            if pd.api.types.is_number(x) and not pd.isna(x): return f"{x:.0f}"
            return ""
        def format_value(x):
            if pd.api.types.is_number(x) and not pd.isna(x):
                if abs(x) < 10 and x != 0: return f"{x:.2f}"
                if abs(x) < 100: return f"{x:.1f}"
                return f"{x:.0f}"
            return ""
        numeric_cols = ["vs. League", "vs. TOP 3"]
        st.markdown("### 🔍 Sekce")
        styler_sec = result["sec_tbl"].style.format(format_percent, na_rep="", subset=numeric_cols).applymap(background_cells, subset=numeric_cols).set_table_styles(table_style_detail_view).hide(axis="index")
        render_styled_df(styler_sec)
        st.markdown("### 🛠️ Podsekce")
        sub_tbl_processed = process_dataframe_for_display(result["sub_tbl"])
        styler_sub = sub_tbl_processed.style.format(format_percent, na_rep="", subset=numeric_cols).applymap(background_cells, subset=numeric_cols).set_table_styles(table_style_detail_view_sub).hide(axis="index")
        render_styled_df(styler_sub)
        st.markdown("### 📋 Všechny metriky")
        styler_all = result["all_metrics"].style.format(format_value, subset=["Hráč", "Liga Ø", "TOP Kluby Ø"], na_rep="").format(format_percent, subset=numeric_cols, na_rep="").applymap(background_cells, subset=numeric_cols).set_table_styles(table_style_detail_view).hide(axis="index")
        render_styled_df(styler_all)
        if st.checkbox("🧠 Zobrazit AI analýzu"):
            if result["gemini_available"]: st.markdown(result["analysis"])
            else: st.warning(result["analysis"])

def page_player_comparison():

    st.markdown("<h1 style='text-align: center;'>Srovnání hráčů napříč soutěžemi</h1>", unsafe_allow_html=True)

    all_players_df = load_all_player_data()
    avg_files = list(Path(AVG_DATA_DIR).glob("*.xlsx"))
    all_avg_dfs = [load_and_process_file(file) for file in avg_files]
    combined_avg_df = pd.concat(all_avg_dfs, ignore_index=True)
    avg_df_filtered = combined_avg_df[combined_avg_df["Minutes played"] >= MIN_MINUTES]
    ratings_df = calculate_ratings_for_all_players(all_players_df, avg_df_filtered)
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

    # --- ZMĚNA ZDE: Nový, citlivější filtr pro tržní hodnotu ---
    st.markdown("---")
    max_val = int(ratings_df['Market value'].max())
    
    # Vytvoříme kroky: do 2M po 50k, od 2M do 5M po 250k, nad 5M po 1M
    steps1 = np.arange(0, 1_000_001, 25000)
    steps2 = np.arange(1_000_000, 5_000_001, 500000)
    steps3 = np.arange(5_000_000, max_val + 1_000_000, 1_000_000)
    
    # Spojíme a odstraníme duplikáty
    market_value_steps = np.unique(np.concatenate((steps1, steps2, steps3))).tolist()

    # Formátovací funkce pro zobrazení hodnot
    def format_market_value(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} M €"
        return f"{value / 1000:.0f} tis. €"

    value_range = st.select_slider(
        "Filtrovat tržní hodnotu:",
        options=market_value_steps,
        value=(market_value_steps[0], market_value_steps[-1]),
        format_func=format_market_value
    )
        
        # Zobrazíme vybraný rozsah v čitelném formátu s oddělovači tisíců
     
    
    # Aplikace filtrů (již bez filtru pro ligu)
    filtered_df = ratings_df
    if selected_pos != "Všechny pozice":
        filtered_df = filtered_df[filtered_df['Position'] == selected_pos]
    if player_search:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(player_search, case=False, na=False)]
    
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1]) &
        (filtered_df['Height'] >= height_range[0]) & (filtered_df['Height'] <= height_range[1]) &
        (filtered_df['Rating vs Liga'] >= rating_range[0]) & (filtered_df['Rating vs Liga'] <= rating_range[1]) &
        (filtered_df['Market value'].between(value_range[0], value_range[1]))
    ]
    
    st.info(f"Zobrazeno {len(filtered_df)} hráčů. Data v tabulce můžete dále řadit a filtrovat.")

    # --- ZMĚNA ZDE: Vlastní JavaScript pro barevný bar chart ---
    bar_chart_renderer = JsCode("""
    class BarChartCellRenderer {
        init(params) {
            this.eGui = document.createElement('div');
            this.eGui.style.width = '100%';
            this.eGui.style.height = '100%';
            this.eGui.style.position = 'relative';
            this.eGui.style.backgroundColor = '#f0f2f6'; // Pozadí prázdné části
            
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
        getGui() {
            return this.eGui;
        }
        refresh(params) {
            const value = params.value;
            if (value === null || value === undefined) {
                this.label.innerHTML = '';
                this.bar.style.width = '0%';
                return true;
            }
            
            let color;
            if (value < 90) { color = '#ffcdd2'; }
            else if (value < 100) { color = '#fff9c4'; }
            else { color = '#c8e6c9'; }
            
            this.bar.style.backgroundColor = color;
            // Škálování baru, předpokládáme maximum 150
            this.bar.style.width = Math.min(100, (value / 150 * 100)) + '%';
            this.label.innerHTML = Math.round(value);
            return true;
        }
    }
    """)
    

    # Odolnější formátovač pro tržní hodnotu
    market_value_formatter = JsCode("""
        function(params) {
            if (params.value !== null && params.value !== undefined && !isNaN(params.value)) {
                return params.value.toLocaleString('cs-CZ', {style: 'currency', currency: 'EUR', maximumFractionDigits: 0});
            } else {
                return '';
            }
        }
    """)

    # --- ZMĚNA ZDE: Vytvoření CSS stylů jako Python slovníku ---
    custom_css = {
        # Obecné pravidlo pro zarovnání všech hlaviček na střed
        ".ag-header-cell-label": {
            "justify-content": "center !important"
        },
        # Specifické pravidlo pro přepsání zarovnání u sloupců Hráč a Tým
        ".ag-header-cell[col-id='Player'] .ag-header-cell-label": {
            "justify-content": "flex-start !important"
        },
        ".ag-header-cell[col-id='Team'] .ag-header-cell-label": {
            "justify-content": "flex-start !important"
        }
    }

    gb = GridOptionsBuilder.from_dataframe(filtered_df)
    
    gb.configure_default_column(sortable=True, filterable=True, resizable=True, wrapHeaderText=True, autoHeaderHeight=True, suppressMenu=True)
    center_aligned_style = {'textAlign': 'center'}

    gb.configure_column("Player", headerName="Hráč", width=200, filter='agSetColumnFilter')
    gb.configure_column("Team", headerName="Tým", width=150)
    gb.configure_column("League", headerName="Soutěž", cellStyle=center_aligned_style, width=150, filter='agSetColumnFilter')
    gb.configure_column("Position", headerName="Pozice", cellStyle=center_aligned_style, width=100,filter=False)
    gb.configure_column("Age", headerName="Věk", cellStyle=center_aligned_style, width=80,filter=False)
    gb.configure_column("Height", headerName="Výška", cellStyle=center_aligned_style, width=80,filter=False)
    
    gb.configure_column(
        "Market value", 
        headerName="Tržní hodnota", 
        valueFormatter=market_value_formatter, 
        cellStyle=center_aligned_style, 
        width=150,
        filter=False
    )
    
    # Použití nového bar chart rendereru
    gb.configure_column("Rating vs Liga", cellRenderer=bar_chart_renderer, width=150,filter=False)
    gb.configure_column("Rating vs TOP Kluby", cellRenderer=bar_chart_renderer, width=180,filter=False)
    
    gridOptions = gb.build()

    AgGrid(
        filtered_df,
        gridOptions=gridOptions,
        height=900,
        width='100%',
        fit_columns_on_grid_load=False,
        theme='Balham',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        custom_css=custom_css
    )

# --- Hlavní APLIKACE s navigací ---
st.set_page_config(page_title="Skautingový report", page_icon="/Users/krejda/Documents/Python/Aplikace/player_app/logo.png", layout="wide")

# --- ZMĚNA ZDE: Vytvoření společné hlavičky ---
# Tento kód se nyní provede vždy, bez ohledu na vybranou stránku
left_col, right_col = st.columns([4,  1])

with right_col:
    st.image("logo.png", width=500) # Zde můžete mít jakékoliv logo aplikace

st.sidebar.title("Navigace")
app_mode = st.sidebar.radio("Zvolte pohled:", ["Srovnání hráčů", "Detail hráče"])

# Zobrazení vybrané stránky
if app_mode == "Detail hráče":
    page_single_player_view()
elif app_mode == "Srovnání hráčů":
    page_player_comparison()
