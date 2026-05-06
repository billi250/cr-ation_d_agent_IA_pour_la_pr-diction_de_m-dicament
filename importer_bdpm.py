from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "bdpm_raw"
OUTPUT_PATH = DATA_DIR / "medicaments_corpus_bdpm.json"

CIS_ZIP_NAMES = ["CIS_bdpm.zip", "CIS_bdpm_officielle.zip", "cis-bdpm-officielle.zip"]
COMPO_ZIP_NAMES = ["CIS_COMPO_bdpm.zip", "cis-compo-bdpm.zip"]

DEFAULT_TERMS = [
    "doliprane", "dafalgan", "efferalgan", "paracétamol", "paracetamol",
    "ibuprofène", "ibuprofene", "advil", "nurofen",
    "aspirine", "aspegic", "aspégic",
    "amoxicilline", "augmentin",
    "smecta", "imodium",
    "ventoline", "becotide", "béclométasone",
    "oméprazole", "omeprazole", "inexium", "esomeprazole",
    "metformine", "glucophage",
    "maxilase", "alpha-amylase",
    "voltarene", "voltarène", "voltaren", "diclofenac", "diclofénac",
    "spasfon", "phloroglucinol",
    "tramadol", "cetirizine", "cétirizine", "levothyrox",
]

SECTION_PATTERNS = {
    "1 Dénomination du médicament": [r"denomination du medicament"],
    "2 Composition qualitative et quantitative": [r"composition qualitative et quantitative", r"composition"],
    "3 Forme pharmaceutique": [r"forme pharmaceutique"],
    "4.1 Indications thérapeutiques": [r"indications therapeutiques", r"indication therapeutique"],
    "4.2 Posologie et mode d'administration": [r"posologie et mode d.?administration", r"posologie"],
    "4.3 Contre-indications": [r"contre.?indications?"],
    "4.4 Mises en garde et précautions d'emploi": [r"mises en garde.*precautions d.?emploi", r"mises en garde", r"precautions d.?emploi"],
    "4.5 Interactions avec d'autres médicaments": [r"interactions avec d.?autres medicaments", r"interactions"],
    "4.6 Fertilité, grossesse et allaitement": [r"fertilite.*grossesse.*allaitement", r"grossesse.*allaitement"],
    "4.7 Effets sur l'aptitude à conduire": [r"effets sur l.?aptitude.*conduire", r"aptitude.*conduire"],
    "4.8 Effets indésirables": [r"effets indesirables", r"effet indesirable", r"effets secondaires"],
    "4.9 Surdosage": [r"surdosage"],
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def extract_zip_if_needed(zip_path: Path) -> list[Path]:
    if not zip_path.exists():
        return []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    print(f"Décompression : {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    return list(dest.rglob("*.txt")) + list(dest.rglob("*.csv"))


def collect_files_from_zips(zip_names: list[str]) -> list[Path]:
    paths: list[Path] = []
    for zip_name in zip_names:
        paths.extend(extract_zip_if_needed(DATA_DIR / zip_name))
    return paths


def find_file(paths: list[Path], contains: str) -> Path | None:
    contains_norm = strip_accents(contains)
    for p in paths:
        if contains_norm in strip_accents(p.name):
            return p
    return paths[0] if paths else None


def detect_table_params(path: Path, nrows: int = 1000) -> tuple[str, str]:
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    separators = ["\t", ";", ","]
    best_score = -1
    best_params = ("latin-1", "\t")
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, header=None, dtype=str,
                                 keep_default_na=False, quoting=csv.QUOTE_NONE,
                                 engine="python", on_bad_lines="skip", nrows=nrows)
                if df.empty:
                    continue
                first_col = df.iloc[:, 0].astype(str)
                numeric_cis_count = first_col.str.match(r"^\d{8}$").sum()
                score = df.shape[1] * 1000 + numeric_cis_count
                if score > best_score:
                    best_score = score
                    best_params = (enc, sep)
            except Exception:
                continue
    enc, sep = best_params
    print(f"Paramètres détectés pour {path.name} : encoding={enc} | sep={'TAB' if sep == chr(9) else sep}")
    return best_params


def read_table(path: Path) -> pd.DataFrame:
    enc, sep = detect_table_params(path)
    df = pd.read_csv(path, sep=sep, encoding=enc, header=None, dtype=str,
                     keep_default_na=False, quoting=csv.QUOTE_NONE,
                     engine="python", on_bad_lines="skip")
    print(f"Lecture table : {path.name} | shape={df.shape}")
    return df


def load_denominations() -> dict[str, dict[str, str]]:
    paths = collect_files_from_zips(CIS_ZIP_NAMES)
    if not paths:
        paths = list(DATA_DIR.glob("CIS_bdpm*.txt")) + list(DATA_DIR.glob("CIS_bdpm*.csv")) + list(DATA_DIR.glob("cis-bdpm*.txt")) + list(DATA_DIR.glob("cis-bdpm*.csv"))
    path = find_file(paths, "CIS_bdpm") or find_file(paths, "cis-bdpm")
    if not path:
        raise FileNotFoundError("Fichier CIS_bdpm introuvable dans data/.")
    df = read_table(path)
    meds: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        cis = str(row.iloc[0]).strip()
        if not re.fullmatch(r"\d{8}", cis):
            continue
        meds[cis] = {
            "cis": cis,
            "medicament": normalize_text(row.iloc[1]) if len(row) > 1 else f"Médicament CIS {cis}",
            "forme": normalize_text(row.iloc[2]) if len(row) > 2 else "",
            "voie": normalize_text(row.iloc[3]) if len(row) > 3 else "",
            "statut": normalize_text(row.iloc[6]) if len(row) > 6 else "",
            "laboratoire": normalize_text(row.iloc[10]) if len(row) > 10 else "",
        }
    print(f"Dénominations chargées : {len(meds)} médicaments")
    return meds


def load_substances() -> dict[str, str]:
    paths = collect_files_from_zips(COMPO_ZIP_NAMES)
    if not paths:
        paths = list(DATA_DIR.glob("CIS_COMPO*.txt")) + list(DATA_DIR.glob("CIS_COMPO*.csv")) + list(DATA_DIR.glob("cis-compo*.txt")) + list(DATA_DIR.glob("cis-compo*.csv"))
    path = find_file(paths, "CIS_COMPO") or find_file(paths, "cis-compo")
    if not path:
        print("Fichier CIS_COMPO introuvable : substances non enrichies.")
        return {}
    df = read_table(path)
    substances: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        cis = str(row.iloc[0]).strip()
        substance = normalize_text(row.iloc[3]) if len(row) > 3 else ""
        nature = str(row.iloc[6]).strip().upper() if len(row) > 6 else ""
        if re.fullmatch(r"\d{8}", cis) and substance and (not nature or nature in {"SA", "ST"}):
            substances.setdefault(cis, set()).add(substance)
    result = {cis: ", ".join(sorted(vals)) for cis, vals in substances.items()}
    print(f"Substances chargées : {len(result)} médicaments")
    return result


def select_medicaments(meds: dict[str, dict[str, str]], substances: dict[str, str], query: str | None, limit: int, max_per_term: int) -> list[str]:
    terms = [query] if query else DEFAULT_TERMS
    selected: list[str] = []
    seen: set[str] = set()
    searchable_by_cis = {cis: strip_accents(info.get("medicament", "") + " " + substances.get(cis, "")) for cis, info in meds.items()}
    for term in terms:
        term_norm = strip_accents(term)
        count = 0
        for cis, s in searchable_by_cis.items():
            if cis in seen:
                continue
            if term_norm in s:
                selected.append(cis)
                seen.add(cis)
                count += 1
                if max_per_term and count >= max_per_term:
                    break
                if limit and len(selected) >= limit:
                    return selected
    return selected[:limit] if limit else selected


def fetch_rcp_text(cis: str, timeout: int = 15) -> str:
    urls = [
        f"https://m.base-donnees-publique.medicaments.gouv.fr/rcp-{cis}-0",
        f"https://base-donnees-publique.medicaments.gouv.fr/medicament/{cis}/rcp",
        f"https://base-donnees-publique.medicaments.gouv.fr/medicament/{cis}/extrait",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (RAG TP educational project)"}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code != 200 or len(resp.text) < 500:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text("\n")
            lines = [normalize_text(line) for line in text.splitlines()]
            lines = [line for line in lines if line]
            text = normalize_text("\n".join(lines))
            if any(strip_accents(k) in strip_accents(text) for k in ["effets indesirables", "posologie", "contre-indications", "composition"]):
                return text
        except Exception:
            continue
    return ""


def split_sections(text: str) -> dict[str, str]:
    text = normalize_text(text)
    if not text:
        return {}
    text_norm = strip_accents(text)
    found: list[tuple[int, str]] = []
    for section_name, patterns in SECTION_PATTERNS.items():
        best = None
        for pat in patterns:
            m = re.search(pat, text_norm, flags=re.IGNORECASE)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            found.append((best, section_name))
    found.sort(key=lambda x: x[0])
    sections: dict[str, str] = {}
    for i, (start, section_name) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) >= 80:
            sections[section_name] = section_text[:25000]
    if not sections:
        sections["Informations générales"] = text[:12000]
    return sections


def build_corpus(query: str | None, limit: int, max_per_term: int, sleep: float) -> list[dict[str, Any]]:
    DATA_DIR.mkdir(exist_ok=True)
    meds = load_denominations()
    substances = load_substances()
    selected = select_medicaments(meds, substances, query, limit, max_per_term)
    if not selected:
        raise RuntimeError("Aucun médicament sélectionné. Essaie --query amoxicilline ou vérifie CIS_bdpm.")
    print(f"Médicaments sélectionnés : {len(selected)}")
    corpus: list[dict[str, Any]] = []
    for cis in tqdm(selected, desc="Téléchargement RCP officiels"):
        info = meds[cis]
        rcp_text = fetch_rcp_text(cis)
        if sleep:
            time.sleep(sleep)
        sections = split_sections(rcp_text)
        # Toujours ajouter un document d'informations générales pour les questions "c'est quoi".
        general = (
            f"Médicament : {info['medicament']}\n"
            f"Code CIS : {cis}\n"
            f"Substance active : {substances.get(cis, 'Non renseignée')}\n"
            f"Forme pharmaceutique : {info.get('forme', '')}\n"
            f"Voie d'administration : {info.get('voie', '')}\n"
            f"Statut : {info.get('statut', '')}\n"
            f"Laboratoire : {info.get('laboratoire', '')}\n"
        )
        sections = {"Informations générales": general, **sections}
        corpus.append({
            "id": f"bdpm_{cis}",
            "cis": cis,
            "medicament": info["medicament"],
            "substance": substances.get(cis, "Non renseignée"),
            "source": "BDPM officielle + RCP en ligne",
            "sections": sections,
        })
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Filtre : doliprane, amoxicilline, maxilase...")
    parser.add_argument("--limit", type=int, default=300, help="Nombre total max de médicaments. 0 = tous les sélectionnés.")
    parser.add_argument("--max-per-term", type=int, default=8, help="Nombre max de spécialités par terme par défaut.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Pause entre requêtes web.")
    args = parser.parse_args()
    corpus = build_corpus(args.query, args.limit, args.max_per_term, args.sleep)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print("\nCorpus BDPM généré avec succès")
    print(f"Fichier créé : {OUTPUT_PATH}")
    print(f"Nombre de médicaments : {len(corpus)}")
    print("Prochaine étape : python indexation.py --corpus data/medicaments_corpus_bdpm.json")


if __name__ == "__main__":
    main()
