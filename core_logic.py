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
    """Sestaví textový prompt pro Gemini AI."""
    return f"""Jsi špičkový fotbalový analytik specializující se na hodnocení hráčů pro účely skautingu a přestupů.
Tvým úkolem je vytvořit hloubkovou textovou analýzu výkonu hráče {player} (pozice: {', '.join(positions)}).

Celkové hodnocení hráče bylo provedeno interně na základě komplexního porovnání s průměrem ligy a TOP 3 týmů.

1.  Celkové vážené skóre:
    -   Vs. Liga: {score_lg:.1f} %
    -   Vs. TOP 3: {score_tp:.1f} %

2.  Porovnání metrik:
{df_all_formatted.to_string(index=False)}

NEZMIŇUJ číselné hodnoty metrik, pouze kvalitativně popiš výkon hráče. Zaměř se na implicitní vlastnosti (např. čtení hry, pracovitost) a přínos pro nový tým.

Struktura výstupu:
- Shrnutí
- Rozbor výkonu
- Implicitní vlastnosti
- Potenciál pro tým
- Doporučení


"""