# core_logic.py
import pandas as pd
from pathlib import Path
import os
import json
from typing import Any, Optional

"""Sjednocená konfigurace pro všechny moduly využívající Gemini."""
GEMINI_PROJECT_ID: str = os.environ.get("GEMINI_PROJECT_ID", "inside-data-story")
GEMINI_LOCATION: str = os.environ.get("GEMINI_LOCATION", "us-central1")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

# Sjednocené názvy pro zdroje tajemství (ENV / Streamlit / lokální soubor)
ENV_SECRET_NAME: str = os.environ.get("GCP_SA_ENV_NAME", "GCP_SA_JSON")
STREAMLIT_SECRET_NAME: str = os.environ.get("STREAMLIT_SA_SECRET_NAME", "gcp_service_account")
LOCAL_SECRET_PATH: str = os.environ.get("GCP_SA_LOCAL_PATH", "inside-data-story-7ffa725c1408.json")

def get_positions_for_avg_filter(main_position: str) -> list[str]:
    
    """
    Vrátí seznam pozic pro filtrování průměrných dat.
    Pro krajní obránce (DR/DL) a krajní záložníky (AMR/AML) vrací obě pozice,
    aby se průměr počítal ze širší skupiny srovnatelných hráčů.
    """
    
    if main_position in ["DR", "DL"]:
        return ["DR", "DL"]
    # Podpora sloučené kategorie křídel ARML (RW/LW/RAMF/LAMF apod.)
    if main_position == "ARML" or main_position in ["AMR", "AML"]:
        return ["AMR", "AML"]
    return [main_position]

def _map_position(pos: str, keys: list) -> Optional[str]:
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


# =============================
# AI Analýza
# =============================


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

# =============================
# AI Skaut
# =============================

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

# =============================
# AI Head to Head
# =============================

def build_head_to_head_prompt(
    p1_name: str,
    p2_name: str,
    header1: dict,
    header2: dict,
    sec_delta_df: pd.DataFrame,
    sub_delta_df: pd.DataFrame,
    top_metric_diffs: pd.DataFrame
) -> str:
    def _fmt_header(h: dict) -> str:
        return (
            f"- Klub: {h.get('Team','N/A')}\n"
            f"- Pozice: {h.get('Position','N/A')}\n"
            f"- Věk: {int(h.get('Age',0))}\n"
            f"- Výška: {int(h.get('Height',0))} cm\n"
            f"- Minuty: {int(h.get('Minutes',0))}\n"
        )

    lines = []
    lines.append("Porovnej dva hráče a napiš scoutingové H2H posouzení ve 4 částech:")
    lines.append("1) Shrnutí (2–3 věty) – kdo je aktuálně vhodnější pro top úroveň a proč.")
    lines.append("2) Taktický fit – v jaké roli/systému by každý hráč exceloval.")
    lines.append("3) Klíčové rozdíly – uveď 5 největších datových rozdílů (pozitivní i negativní).")
    lines.append("4) Rizika & doporučení – co může selhat, jaké jsou limity dat.\n")

    lines.append(f"=== Hráč A: {p1_name} ===")
    lines.append(_fmt_header(header1))
    lines.append(f"=== Hráč B: {p2_name} ===")
    lines.append(_fmt_header(header2))

    def _tbl(df: pd.DataFrame, head: str, cols: list[str], limit: int = 20) -> list[str]:
        out = [head]
        if df is None or df.empty:
            out.append("- (žádná srovnatelná data)")
            return out
        dfx = df.copy()
        for c in cols:
            if c in dfx.columns:
                dfx[c] = pd.to_numeric(dfx[c], errors='coerce')
        dfx = dfx.head(limit)
        for _, r in dfx.iterrows():
            key = r.get('Section', r.iloc[0])
            vals = ", ".join([f"{c}={r.get(c)}" for c in cols if c in dfx.columns])
            out.append(f"- {key}: {vals}")
        return out

    sec_cols = ["P1 vs. Liga", "P2 vs. Liga", "Δ vs. Liga", "P1 vs. TOP 3", "P2 vs. TOP 3", "Δ vs. TOP 3"]
    lines += _tbl(sec_delta_df, "\n[SEKCE – rozdíly]", sec_cols, limit=30)

    sub_cols = ["P1 vs. Liga", "P2 vs. Liga", "Δ vs. Liga"]
    if sub_delta_df is not None and not sub_delta_df.empty and "Δ vs. Liga" in sub_delta_df.columns:
        sub_sorted = sub_delta_df.assign(abs_delta=sub_delta_df["Δ vs. Liga"].abs()).sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"])
    else:
        sub_sorted = sub_delta_df
    lines += _tbl(sub_sorted, "\n[PODSEKCE – top rozdíly]", sub_cols, limit=25)

    if top_metric_diffs is not None and not top_metric_diffs.empty:
        lines.append("\n[TOP METRIKY – největší rozdíly (vs. liga)]")
        for _, r in top_metric_diffs.iterrows():
            try:
                lines.append(f"- {r['Metric']}: {p1_name} {float(r['P1 vs. League']):.0f}% vs {p2_name} {float(r['P2 vs. League']):.0f}% (Δ={float(r['DeltaAbs']):.0f} p.b.)")
            except Exception:
                continue
    else:
        lines.append("\n[TOP METRIKY – žádná data]")

    lines.append("\nPiš stručně, česky, bez superlativů. Zahrň nejistoty (vzorek minut apod.).")
    return "\n".join(lines)


# =============================
# Unified Gemini initialization
# =============================

def initialize_gemini_shared(
    project_id: str,
    location: str,
    model_name: str,
    env_secret_name: str = "GCP_SA_JSON",
    streamlit_secret_name: str = "gcp_service_account",
    local_secret_path: str = "inside-data-story-7ffa725c1408.json",
):
    """Jednotná inicializace Vertex AI Gemini pro všechny moduly.

    Zdroj přihlašovacích údajů (v tomto pořadí):
    1) ENV proměnná s JSON obsahem (např. Secret Manager → env)
    2) Lokální JSON soubor (mimo git)
    3) Streamlit `st.secrets`

    Vrací tuple (model, available). Při chybě vrací (None, False).
    """
    # Importy pouze lokálně, aby se nenačetly zbytečně při importu modulu
    try:
        import streamlit as st  # type: ignore
        _has_streamlit = True
    except Exception:  # pragma: no cover
        _has_streamlit = False

    try:
        from google.oauth2 import service_account  # type: ignore
        import vertexai  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
    except Exception:
        if _has_streamlit:
            st.error("Chybí balíčky google-cloud-aiplatform / google-auth. Nainstalujte je podle requirements.txt.")
        else:
            print("Chybí balíčky google-cloud-aiplatform / google-auth.")
        return None, False

    creds = None
    secret_info: Optional[Any] = None

    env_val = os.environ.get(env_secret_name)
    if env_val:
        try:
            secret_info = json.loads(env_val)
            print("--- INFO: Nalezen klíč v proměnné prostředí (ENV). ---")
        except json.JSONDecodeError as e:
            if _has_streamlit:
                st.error(f"Chyba při parsování JSON z proměnné prostředí '{env_secret_name}': {e}")
            else:
                print(f"Chyba při parsování JSON z ENV: {e}")
            return None, False
    # Alternativa: ENV s cestou k souboru (snadné pro lokální běh)
    elif os.environ.get("GCP_SA_JSON_PATH") and os.path.exists(os.environ.get("GCP_SA_JSON_PATH", "")):
        try:
            with open(os.environ.get("GCP_SA_JSON_PATH", "")) as f:
                secret_info = json.load(f)
            print("--- INFO: Nalezen klíč podle ENV GCP_SA_JSON_PATH. ---")
        except Exception as e:
            if _has_streamlit:
                st.error(f"Chyba při čtení souboru definovaného v GCP_SA_JSON_PATH: {e}")
            else:
                print(f"Chyba při čtení souboru z GCP_SA_JSON_PATH: {e}")
            return None, False
    # Další běžná proměnná – GOOGLE_APPLICATION_CREDENTIALS (cesta k souboru)
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")):
        try:
            with open(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")) as f:
                secret_info = json.load(f)
            print("--- INFO: Nalezen klíč podle ENV GOOGLE_APPLICATION_CREDENTIALS. ---")
        except Exception as e:
            if _has_streamlit:
                st.error(f"Chyba při čtení souboru z GOOGLE_APPLICATION_CREDENTIALS: {e}")
            else:
                print(f"Chyba při čtení souboru z GOOGLE_APPLICATION_CREDENTIALS: {e}")
            return None, False
    elif os.path.exists(local_secret_path):
        try:
            with open(local_secret_path) as f:
                secret_info = json.load(f)
            print(f"--- INFO: Nalezen klíč v lokálním souboru '{local_secret_path}'. ---")
        except Exception as e:
            if _has_streamlit:
                st.error(f"Chyba při čtení lokálního souboru s klíčem: {e}")
            else:
                print(f"Chyba při čtení lokálního souboru s klíčem: {e}")
            return None, False
    elif _has_streamlit and hasattr(st, "secrets"):
        try:
            # Může vyhodit FileNotFoundError, pokud secrets.toml neexistuje – ošetříme a pokračujeme.
            if streamlit_secret_name in st.secrets:
                secret_info = dict(st.secrets[streamlit_secret_name])
                print("--- INFO: Nalezen klíč ve Streamlit Secrets. ---")
        except FileNotFoundError:
            # Žádné secrets – ignorujeme a pokračujeme k další větvi
            pass
        except Exception as e:
            if _has_streamlit:
                st.error(f"Chyba při čtení Streamlit secrets: {e}")
            else:
                print(f"Chyba při čtení Streamlit secrets: {e}")
            return None, False

    if secret_info:
        try:
            creds = service_account.Credentials.from_service_account_info(secret_info)  # type: ignore
        except Exception as e:
            if _has_streamlit:
                st.error(f"Chyba při vytváření přihlašovacích údajů z nalezeného klíče: {e}")
            else:
                print(f"Chyba při vytváření přihlašovacích údajů: {e}")
            return None, False
    else:
        msg = (
            "Chybí přihlašovací údaje pro Google Cloud! Zkontrolujte nastavení pro vaše prostředí:\n\n"
            f"- Pro Google Cloud Run: nastavte secret v ENV jako '{env_secret_name}'.\n"
            f"- Pro lokální spuštění: ujistěte se, že existuje soubor '{local_secret_path}'.\n"
            f"- Pro Streamlit Cloud: přidejte secret s názvem '{streamlit_secret_name}'."
        )
        if _has_streamlit:
            st.error(msg)
        else:
            print(msg)
        return None, False

    try:
        vertexai.init(project=project_id, location=location, credentials=creds)  # type: ignore
        model = GenerativeModel(model_name)  # type: ignore
        print("--- INFO: Vertex AI úspěšně inicializováno (shared). ---")
        return model, True
    except Exception as e:
        warn = f"Klíč byl načten, ale selhala inicializace Vertex AI: {e}"
        if _has_streamlit:
            st.warning(warn)
        else:
            print(warn)
        return None, False