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


####### AI Analýza


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

#AI Skaut

def build_ai_scout_prompt(needs_description: str, players_data_string: str) -> str:
    """Sestaví finální, expertní prompt pro AI Skauta, který hledá hráče podle volného textu."""
    
    return f"""Jsi elitní fotbalový skaut s dokonalou schopností analyzovat data a porozumět potřebám klubu. Tvým úkolem je na základě textového zadání najít nejvhodnější hráče.

**POŽADAVKY SPORTOVNÍHO ŘEDITELE (NAPSÁNO VOLNÝM TEXTEM):**
---
"{needs_description}"
---

**SEZNAM DOSTUPNÝCH HRÁČŮ S JEJICH KLÍČOVÝMI DATY A HODNOCENÍM:**
(Hodnocení je vždy ve formátu "Název: Rating vs Liga / Rating vs TOP Kluby". 100 = průměr.)
---
{players_data_string}
---

**TVŮJ ÚKOL - POSTUPUJ PŘESNĚ PODLE TĚCHTO KROKŮ:**

**1. DEŠIFRUJ POŽADAVKY:**
Pečlivě si přečti textové zadání. Identifikuj všechny klíčové vlastnosti a kritéria, které ředitel hledá (např. "kreativní", "dominantní v hlavičkových soubojích", "mladý s potenciálem", "dobrý v pressingu", "levný", "ze slovenské ligy").

**2. PROPOJ POŽADAVKY S DATY (NEJDŮLEŽITĚJŠÍ KROK):**
Pro každou požadovanou vlastnost si v hlavě urči, které **sekce a podsekce** v datech jsou nejdůležitější. Jednej jako skutečný expert. Zde je návod, jak uvažovat:

* **FYZICKÉ A ZÁKLADNÍ ÚDAJE:**
    * "Mladý", "perspektivní", "do 23 let" -> Hledej nízký `Věk`.
    * "Vysoký", "dobrá postava" -> Hledej vysokou `Výška`.
    * "Levonohý", "pravonohý" -> Hledej konkrétní hodnotu v `Noha`.
    * "Levný", "dostupný", "nízká cena" -> Hledej nízkou `Tržní hodnota`.
    * "Ze slovenské ligy" -> Hledej pouze hráče, kde `Soutěž` obsahuje "Slovakia".

* **DEFENZIVNÍ VLASTNOSTI:**
    * "Dominantní v hlavičkových soubojích" -> Hledej vysoké hodnocení v podsekci `Kvalita ve vzduchu`.
    * "Dobrý v odebírání míče", "skvělý v soubojích 1 na 1" -> Hledej vysoké hodnocení v podsekci `Soubojová kvalita na zemi`.
    * "Pracovitý", "dobrý v pressingu", "aktivní bez míče" -> Hledej vysoké hodnocení v podsekci `Soubojová intezita na zemi`.

* **OFENZIVNÍ VLASTNOSTI A KREATIVITA:**
    * "Kreativní", "má finální přihrávku", "tvoří šance" -> Hledej vysoké hodnocení v podsekcích `Příprava šancí` a `Nadstavbové přihrávky kvalita`.
    * "Skvělý střelec", "gólový" -> Hledej vysoké hodnocení v podsekci `Zakončení`.
    * "Dobrý driblér", "umí obejít hráče" -> Hledej vysoké hodnocení v sekci `Dostat se přes hráče` a podsekci `Úspěšnost dostat se přes hráče`.
    * "Kvalitní rozehrávka", "přesné přihrávky" -> Hledej vysoké hodnocení v sekci `Přihrávky` a podsekci `Základní přihrávky kvalita`.

**3. ANALYZUJ A VYBER:**
Projdi seznam hráčů a na základě svého expertního úsudku najdi **5 až 10 nejlepších kandidátů**, kteří se nejvíce blíží popsanému ideálnímu profilu. Nehledej jen přesnou shodu, ale hráče, kteří nejlépe splňují kombinaci nejdůležitějších požadavků.

**4. ZDŮVODNI SVŮJ VÝBĚR:**
Pro každého doporučeného hráče napiš krátké (2-3 věty) a výstižné zdůvodnění. Vždy se odkaž na **konkrétní sekce nebo podsekce**, ve kterých hráč vyniká a které odpovídají požadavkům. Příklad: "Doporučuji hráče X, protože dle dat exceluje v podsekci Kvalita ve vzduchu (125/120), což přesně odpovídá našemu požadavku na dominantního hlavičkáře."

**FORMÁT VÝSTUPU:**
Začni krátkým shrnutím, jak jsi pochopil zadání. Poté vypiš doporučené hráče s odůvodněním ve formátu odrážek.
"""