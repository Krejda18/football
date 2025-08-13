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

def build_prompt(player: str, positions: list, sec_tbl: pd.DataFrame, sub_tbl: pd.DataFrame, all_metrics_tbl: pd.DataFrame) -> str:
    """Sestaví vylepšený, komplexní a vyvážený textový prompt pro Gemini AI."""

    # Převedeme tabulky na přehledný textový formát pro prompt
    sec_tbl_str = sec_tbl[['Section', 'vs. League', 'vs. TOP 3']].to_string(index=False, float_format="%.0f")
    sub_tbl_str = sub_tbl[['Section', 'Subsection', 'vs. League', 'vs. TOP 3']].to_string(index=False, float_format="%.0f")
    all_metrics_str = all_metrics_tbl[['Metric', 'Hráč', 'Liga Ø', 'vs. League', 'vs. TOP 3']].to_string(index=False)


    return f"""Jsi špičkový, kritický fotbalový analytik specializující se na objektivní hodnocení hráčů.
Tvým úkolem je vytvořit hloubkovou, vyváženou textovou analýzu výkonu hráče {player} (pozice: {', '.join(positions)}).

Tvá analýza MUSÍ vycházet ze VŠECH poskytnutých datových sad a propojovat je:
1. Agregované hodnocení v hlavních oblastech (Sekce)
2. Detailnější hodnocení v dílčích dovednostech (Podsekce)
3. Hodnoty jednotlivých metrik

--- DATA PRO ANALÝZU ---

1. HODNOCENÍ V SEKCIÍCH (100 = průměr):
{sec_tbl_str}

2. HODNOCENÍ V PODSEKCÍCH (100 = průměr):
{sub_tbl_str}

3. DETAILNÍ METRIKY HRÁČE A SROVNÁVACÍ PRŮMĚRY:
{all_metrics_str}

--- DŮLEŽITÉ POKYNY ---
- **Syntéza, ne jen popis:** Začni od obecného (hodnocení v sekcích) a vysvětli ho pomocí konkrétních podsekcí a detailních metrik.
- **Vyváženost:** Aktivně hledej a zmiň jak **silné stránky** (hodnoty nad 100), tak **slabiny** (hodnoty pod 100). Analýza musí být objektivní.
- **Kvalitativní popis:** NEZMIŇUJ explicitní číselné hodnoty z tabulek. Popisuj výkon kvalitativně (např. "výrazně nadprůměrný", "zaostává", "průměrný").
- **Porovnání s TOP 3:** Při hodnocení **vždy porovnávej výkon hráče jak s průměrem celé ligy, tak s průměrem TOP 3 klubů.** Srovnání s TOP 3 je klíčové pro posouzení jeho potenciálu pro přestup do špičkového týmu. Zhodnoť, zda hráč v klíčových dovednostech dosahuje, nebo naopak zaostává za úrovní TOP 3.
- **Implicitní vlastnosti:** Na základě dat odvoď vlastnosti jako čtení hry, pracovitost, efektivita pod tlakem atd.

--- POŽADOVANÁ STRUKTURA VÝSTUPU ---
- **Celkové shrnutí a profil hráče:** Vyvážený pohled na jeho styl.
- **Klíčové silné stránky:** 2-3 nejdůležitější přednosti podložené srovnáním s ligou i TOP 3.
- **Oblasti pro zlepšení / Slabiny:** 2-3 největší slabiny, zejména ve srovnání s úrovní TOP 3.
- **Taktické využití a potenciál pro tým:** Jak by mohl být pro tým přínosem.
- **Závěrečné doporučení:** Stručné shrnutí s ohledem na jeho potenciál pro TOP kluby.
"""
