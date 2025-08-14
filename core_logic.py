# core_logic.py
import pandas as pd
from pathlib import Path

def get_positions_for_avg_filter(main_position: str) -> list[str]:
    """
    Vrátí seznam pozic pro filtrování průměrných dat.
    Pro krajní obránce (DR/DL) a krajní záložníky (AMR/AML) vrací obě pozice,
    aby se průměr počítal ze širší skupiny srovnatelných hráčů.
    """
    if main_position in ["DR", "DL"]:
        return ["DR", "DL"]
    if main_position in ["AMR", "AML"]:
        return ["AMR", "AML"]
    return [main_position]

def compute_loose_ball_duels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = ["Duels per 90", "Defensive duels per 90", "Offensive duels per 90", "Aerial duels per 90", "Duels won", "Defensive duels won", "Offensive duels won", "Aerial duels won"]
    if not all(col in df.columns for col in required_cols):
        return df
    try:
        df["Loose ball duels per 90"] = df["Duels per 90"] - (df["Defensive duels per 90"] + df["Offensive duels per 90"] + df["Aerial duels per 90"])
        loose_won_raw = (df["Duels per 90"] * df["Duels won"] / 100) - (df["Defensive duels per 90"] * df["Defensive duels won"] / 100 + df["Offensive duels per 90"] * df["Offensive duels won"] / 100 + df["Aerial duels per 90"] * df["Aerial duels won"] / 100)
        df["Loose ball duels won"] = (loose_won_raw / df["Loose ball duels per 90"].replace(0, pd.NA)) * 100
    except Exception:
        pass
    return df

def compute_effective_passing_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = ["Average pass length", "Accurate passes"]
    if not all(col in df.columns for col in required_cols):
        return df
    try:
        df["Effective Passing Index"] = (df["Average pass length"] * df["Accurate passes"]) / 1000
    except Exception:
        pass
    return df

def rating_series(player: pd.Series, base: pd.Series) -> pd.Series:
    idx = player.index.intersection(base.index)
    return (player[idx] / base[idx].replace(0, pd.NA) * 100).replace([float("inf"), -float("inf")], pd.NA).dropna()

def weighted_score(rats: pd.Series, w_df: pd.DataFrame) -> float:
    m = w_df.set_index("Metric").join(rats.to_frame("R")).dropna()
    if not m.empty and m["Weight"].sum() > 0:
        return float((m["R"] * m["Weight"]).sum() / m["Weight"].sum())
    return float("nan")

def _map_position(pos: str, keys: list) -> str or None:
    p = pos.upper()
    if p in keys:
        return p
    mapping = {
        ("AMR", "AML"): "ARML",
        ("DL", "DR"): "DR",
        ("ST", "CF", "FW"): "FC"
    }
    for aliases, target_key in mapping.items():
        if p in aliases and target_key in keys:
            return target_key
    return None

def logic_flat_df(positions: list, logic_data: dict) -> pd.DataFrame:
    recs = []
    for p in positions:
        key = _map_position(p, list(logic_data.keys()))
        if not key:
            continue
        position_logic = logic_data.get(key, {})
        for sec_name, sec in position_logic.items():
            if sec_name == "weight": continue
            w_sec = sec.get("weight", 1.0)
            for sub_name, sub in sec.items():
                if sub_name == "weight": continue
                w_sub = sub.get("weight", 1.0)
                for m, w in sub.items():
                    if m == "weight": continue
                    recs.append({"Metric": m, "Weight": w * w_sub * w_sec})
    if not recs:
        return pd.DataFrame(columns=["Metric", "Weight"])
    return pd.DataFrame(recs).groupby("Metric", as_index=False).agg({"Weight": "mean"})

def breakdown_scores(rats: pd.Series, position: str, logic_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = _map_position(position, list(logic_data.keys()))
    sec_rows, sub_rows = [], []
    if not key:
        return pd.DataFrame(), pd.DataFrame()
    position_logic = logic_data.get(key, {})
    for sec_name, sec in position_logic.items():
        if sec_name == "weight": continue
        sec_score, sec_weight = 0, 0
        for sub_name, sub in sec.items():
            if sub_name == "weight": continue
            w_sub = sub.get("weight", 1.0)
            sub_score, sub_weight = 0, 0
            for m, w in sub.items():
                if m == "weight": continue
                if m in rats.index and pd.notna(rats[m]):
                    sub_score += rats[m] * w
                    sub_weight += w
            if sub_weight > 0:
                sc = sub_score / sub_weight
                sec_score += sc * w_sub
                sec_weight += w_sub
                sub_rows.append({"Section": sec_name.strip(), "Subsection": sub_name.strip(), "Score": sc})
        if sec_weight > 0:
            sec_rows.append({"Section": sec_name.strip(), "Score": sec_score / sec_weight})
    return pd.DataFrame(sec_rows), pd.DataFrame(sub_rows)

def build_prompt(player: str, positions: list, score_lg: float, score_tp: float, df_all_formatted: pd.DataFrame) -> str:
    """
    Sestaví vylepšený, komplexní a vyvážený textový prompt pro Gemini AI,
    který se vyhýbá superlativům a používá původní sadu vstupních parametrů.
    """
    # Převedeme hlavní datovou tabulku na přehledný textový formát pro prompt
    data_str = df_all_formatted.to_string(index=False)

    return f"""Jsi špičkový, kritický fotbalový analytik specializující se na objektivní hodnocení hráčů.
Tvým úkolem je vytvořit hloubkovou, vyváženou textovou analýzu výkonu hráče {player} (pozice: {', '.join(positions)}).

Tvá analýza MUSÍ vycházet ze VŠECH poskytnutých dat a propojovat je. NEZMIŇUJ číselné hodnoty metrik, pouze kvalitativně popiš výkon hráče.

--- DATA PRO ANALÝZU ---

1.  **Celkové vážené skóre (kontext pro tebe, nezmiňuj v analýze):**
    -   Srovnání s průměrem ligy: {score_lg:.1f} %
    -   Srovnání s průměrem TOP 3 týmů: {score_tp:.1f} %

2.  **DETAILNÍ METRIKY A HODNOCENÍ V JEDNOTLIVÝCH OBLASTECH:**
    (Hodnoty "vs. League" a "vs. TOP 3" jsou indexované, kde 100 = průměr)
{data_str}

--- DŮLEŽITÉ POKYNY PRO ZPRACOVÁNÍ ---

- **Syntéza, ne jen popis:** Propojuj data. Pokud například vidíš vysoké hodnocení v sekci "Bránění", vysvětli ho pomocí konkrétních metrik z tabulky, jako jsou "Úspěšnost obranných soubojů" nebo "Získané míče".
- **Vyváženost:** Aktivně hledej a zmiň jak **silné stránky** (hodnoty výrazně nad 100), tak **slabiny** (hodnoty výrazně pod 100). Analýza musí být objektivní.
- **Kvalitativní popis:** NEZMIŇUJ explicitní číselné hodnoty z tabulek (např. "jeho hodnota je 125"). Místo toho popisuj výkon kvalitativně (např. "jeho schopnost vyhrávat souboje je výrazně nadprůměrná", "v rozehrávce zaostává", "jeho presink je na průměrné úrovni").
- **Profesionální a neutrální tón:** Vyhni se přehnaně pozitivním superlativům jako 'vynikající', 'skvělý', 'fenomenální' nebo 'perfektní'. Udržuj jazyk věcný a analytický.
- **Porovnání s TOP 3:** Při hodnocení **vždy porovnávej výkon hráče jak s průměrem celé ligy, tak s průměrem TOP 3 klubů.** Srovnání s TOP 3 je klíčové pro posouzení jeho potenciálu pro přestup do špičkového týmu. Zhodnoť, zda hráč v klíčových dovednostech dosahuje, nebo naopak zaostává za úrovní TOP 3.
- **Odvození implicitních vlastností:** Na základě dat odvoď vlastnosti, které nejsou přímo v tabulce, jako je čtení hry, pracovitost, efektivita pod tlakem, taktická disciplína atd.

--- POŽADOVANÁ STRUKTURA VÝSTUPU ---

**1. Celkové shrnutí a profil hráče:**
Stručný odstavec, který představí hráče a jeho herní styl na základě syntézy všech dat.

**2. Klíčové silné stránky:**
Odrážky se 2-3 nejdůležitějšími přednostmi. Každou podlož konkrétním pozorováním z dat a srovnáním s ligou i TOP 3.

**3. Oblasti pro zlepšení / Slabiny:**
Odrážky se 2-3 největšími slabinami. Uveď, v čem a proč zaostává, zejména ve srovnání s úrovní TOP 3 týmů.

**4. Taktické využití a potenciál pro tým:**
Jak by mohl být pro svůj budoucí tým přínosem? Na jakou pozici a do jakého herního systému by se hodil?

**5. Závěrečné doporučení:**
Stručné shrnutí (1-2 věty) s ohledem na jeho potenciál pro přestup do špičkového klubu.
"""
