
from __future__ import annotations

import io
import json
import os
import re
import textwrap
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration générale
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

INDEX_PATH = STORAGE_DIR / "medicaments.index"
CHUNKS_PATH = STORAGE_DIR / "chunks_medicaments.json"
CONFIG_PATH = STORAGE_DIR / "index_config.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

TOP_K_DEFAULT = 6
SEUIL_CONFIANCE_DEFAULT = 0.35

MENTION_OBLIGATOIRE = "Ces informations ne remplacent pas l'avis d'un professionnel de santé."
BDPM_BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr/medicament/{cis}/extrait"

STOPWORDS_MED = {
    "quelle", "quelles", "quels", "quel", "sont", "est", "les", "des", "une", "dans", "avec", "pour",
    "effet", "effets", "secondaire", "secondaires", "indesirable", "indesirables", "posologie", "dose",
    "prendre", "peut", "puis", "combien", "comment", "medicament", "medicaments", "traitement",
    "risque", "contre", "indication", "indications", "utilisation", "utiliser", "adulte", "enfant",
    "femme", "grossesse", "allaitement", "quoi", "cest", "c", "donne", "explique", "savoir",
    "information", "informations", "faire", "fait", "jai", "j", "sur", "du", "de", "la", "le",
}


# ============================================================
# Fonctions texte
# ============================================================

def sans_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def normaliser_recherche(text: Any) -> str:
    text = sans_accents(str(text))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def termes_importants(question: str) -> list[str]:
    q = sans_accents(question)
    tokens = re.findall(r"[a-z0-9]{4,}", q)
    return [t for t in tokens if t not in STOPWORDS_MED]


def fuzzy_present(term: str, text: str) -> bool:
    text_norm = sans_accents(text)
    term_norm = sans_accents(term)

    if term_norm in text_norm:
        return True

    words = set(re.findall(r"[a-z0-9]{4,}", text_norm))

    for w in words:
        if abs(len(w) - len(term_norm)) <= 3:
            if SequenceMatcher(None, term_norm, w).ratio() >= 0.84:
                return True

    return False


def nettoyer_affichage(text: Any, max_len: int = 160) -> str:
    if text is None:
        return "Non renseigné"

    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)

    text = (
        text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "Non renseigné"

    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."

    return text


def lien_bdpm(cis: str) -> str:
    cis_clean = re.sub(r"\D", "", str(cis))

    if not cis_clean:
        return ""

    return BDPM_BASE_URL.format(cis=cis_clean)


def extraire_noms_medicaments_depuis_question(question: str, noms_disponibles: list[str], limite: int = 4) -> list[str]:
    q_norm = normaliser_recherche(question)
    trouves: list[str] = []

    for nom in noms_disponibles:
        nom_simple = normaliser_recherche(nom)
        if not nom_simple:
            continue

        premiers_mots = " ".join(nom_simple.split()[:2])
        premier_mot = nom_simple.split()[0] if nom_simple.split() else ""

        if premier_mot and len(premier_mot) >= 5 and premier_mot in q_norm:
            trouves.append(nom)
        elif premiers_mots and len(premiers_mots) >= 7 and premiers_mots in q_norm:
            trouves.append(nom)

        if len(trouves) >= limite:
            break

    return list(dict.fromkeys(trouves))


# ============================================================
# Export PDF simple sans dépendance externe
# ============================================================

def _pdf_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", "")
    )


def generer_pdf_simple(titre: str, contenu: str) -> bytes:
    """
    Génère un PDF texte simple sans dépendance externe.
    Suffisant pour exporter la réponse, les sources et les liens.
    """
    largeur_page = 595
    hauteur_page = 842
    marge_x = 45
    marge_y = 55
    ligne_h = 14

    lignes: list[str] = []
    lignes.append(titre)
    lignes.append("=" * 70)
    lignes.extend(str(contenu).splitlines())

    wrapped: list[str] = []
    for ligne in lignes:
        if not ligne.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(ligne, width=92, replace_whitespace=False))

    pages: list[list[str]] = []
    page: list[str] = []
    max_lignes = int((hauteur_page - 2 * marge_y) / ligne_h)

    for ligne in wrapped:
        page.append(ligne)
        if len(page) >= max_lignes:
            pages.append(page)
            page = []

    if page:
        pages.append(page)

    objects: list[str] = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{' '.join(f'{3 + i * 2} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>")

    for i, page_lines in enumerate(pages):
        page_obj_id = 3 + i * 2
        content_obj_id = page_obj_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {largeur_page} {hauteur_page}] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {content_obj_id} 0 R >>"
        )

        stream_lines = ["BT", "/F1 10 Tf", f"{marge_x} {hauteur_page - marge_y} Td"]
        first = True
        for line in page_lines:
            if first:
                first = False
            else:
                stream_lines.append(f"0 -{ligne_h} Td")
            stream_lines.append(f"({_pdf_escape(line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)
        objects.append(f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream")

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]

    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{obj_id} 0 obj\n".encode("latin-1"))
        pdf.write(obj.encode("latin-1", errors="replace"))
        pdf.write(b"\nendobj\n")

    xref_pos = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.write(f"{off:010d} 00000 n \n".encode("latin-1"))

    pdf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1")
    )
    return pdf.getvalue()


def formater_reponse_telechargement(question: str, reponse: str, chunks: list[dict[str, Any]]) -> str:
    lignes = [
        "Assistant RAG Médicaments",
        "=" * 60,
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        "Question :",
        question,
        "",
        "Réponse :",
        reponse,
        "",
        "Sources récupérées :",
    ]

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        medicament = nettoyer_affichage(meta.get("medicament", "Médicament inconnu"), 220)
        section = nettoyer_affichage(meta.get("section", "Section inconnue"), 130)
        cis = nettoyer_affichage(meta.get("cis", "CIS inconnu"), 40)
        substance = nettoyer_affichage(meta.get("substance", "Non renseignée"), 180)
        score = float(chunk.get("score", 0.0))
        url = lien_bdpm(cis)

        lignes.append(f"{i}. {medicament}")
        lignes.append(f"   Section : {section}")
        lignes.append(f"   Substance : {substance}")
        lignes.append(f"   CIS : {cis}")
        lignes.append(f"   Score : {score:.3f}")
        if url:
            lignes.append(f"   Lien : {url}")
        lignes.append("")

    return "\n".join(lignes)


# ============================================================
# Intention de la question
# ============================================================

def intention_question(question: str) -> str:
    q = sans_accents(question)

    if any(mot in q for mot in ["effet", "effets", "indesirable", "indesirables", "secondaire", "secondaires"]):
        return "effets_indesirables"

    if any(mot in q for mot in ["posologie", "dose", "prendre", "administration", "combien"]):
        return "posologie"

    if any(mot in q for mot in ["contre indication", "contre-indication", "interdit", "ne pas prendre", "allergie"]):
        return "contre_indications"

    if any(mot in q for mot in ["interaction", "interactions", "associer", "melanger", "mélanger", "avec"]):
        return "interactions"

    return "general"


def libelle_intention(intention: str) -> str:
    labels = {
        "effets_indesirables": "Effets indésirables",
        "posologie": "Posologie",
        "contre_indications": "Contre-indications",
        "interactions": "Interactions",
        "general": "Information générale",
    }
    return labels.get(intention, "Information générale")


def section_attendue(intention: str) -> str:
    mapping = {
        "effets_indesirables": "4.8",
        "posologie": "4.2",
        "contre_indications": "4.3",
        "interactions": "4.5",
    }
    return mapping.get(intention, "")


def enrichir_question_pour_recherche(question: str) -> str:
    intention = intention_question(question)

    if intention == "effets_indesirables":
        return question + " 4.8 effets indésirables effets secondaires réactions allergiques troubles digestifs"

    if intention == "posologie":
        return question + " 4.2 posologie mode d'administration dose administration"

    if intention == "contre_indications":
        return question + " 4.3 contre-indications hypersensibilité ne doit jamais être utilisé"

    if intention == "interactions":
        return question + " 4.5 interactions avec d'autres médicaments association"

    return question


def mots_cles_intention(intention: str) -> list[str]:
    if intention == "effets_indesirables":
        return [
            "effets indésirables",
            "effet indésirable",
            "effets secondaires",
            "troubles",
            "nausées",
            "vomissements",
            "diarrhée",
            "éruption",
            "allergique",
            "hypersensibilité",
            "anaphylactique",
        ]

    if intention == "posologie":
        return [
            "posologie",
            "mode d'administration",
            "dose",
            "administration",
            "prendre",
            "traitement",
        ]

    if intention == "contre_indications":
        return [
            "contre-indications",
            "contre indications",
            "ne doit jamais être utilisé",
            "hypersensibilité",
            "allergie",
        ]

    if intention == "interactions":
        return [
            "interactions",
            "interactions avec d'autres médicaments",
            "association",
            "médicaments",
        ]

    return []


# ============================================================
# Chargement FAISS / modèle / dictionnaires
# ============================================================

@st.cache_resource(show_spinner=False)
def charger_modele_embedding() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def charger_index_et_chunks() -> tuple[faiss.Index, list[dict[str, Any]]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Base vectorielle introuvable. Vérifie que le dossier storage contient "
            "medicaments.index et chunks_medicaments.json."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


@st.cache_data(show_spinner=False)
def charger_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def construire_catalogue(chunks_avec_meta: list[dict[str, Any]]) -> dict[str, Any]:
    noms: set[str] = set()
    substances: set[str] = set()
    par_medicament: dict[str, list[int]] = {}
    par_substance: dict[str, list[int]] = {}

    for idx, chunk in enumerate(chunks_avec_meta):
        meta = chunk.get("metadata", {})
        medicament = nettoyer_affichage(meta.get("medicament", ""), 500)
        substance = nettoyer_affichage(meta.get("substance", ""), 500)

        if medicament and medicament != "Non renseigné":
            noms.add(medicament)
            key = normaliser_recherche(medicament)
            par_medicament.setdefault(key, []).append(idx)

        if substance and substance != "Non renseigné":
            for sub in re.split(r"[,;/]+", substance):
                sub = nettoyer_affichage(sub.strip(), 200)
                if sub and sub != "Non renseigné":
                    substances.add(sub)
                    key_s = normaliser_recherche(sub)
                    par_substance.setdefault(key_s, []).append(idx)

    return {
        "noms": sorted(noms),
        "substances": sorted(substances),
        "par_medicament": par_medicament,
        "par_substance": par_substance,
    }


# ============================================================
# Recherche FAISS + mots-clés + filtres
# ============================================================

def score_mots_cles(question: str, chunk: dict[str, Any], intention: str) -> float:
    meta = chunk.get("metadata", {})
    texte = normaliser_recherche(
        " ".join(
            [
                str(meta.get("medicament", "")),
                str(meta.get("substance", "")),
                str(meta.get("section", "")),
                str(chunk.get("contenu", ""))[:4500],
            ]
        )
    )

    termes = termes_importants(question)
    if not termes:
        base = 0.0
    else:
        presents = sum(1 for t in termes if normaliser_recherche(t) in texte)
        base = presents / max(len(termes), 1)

    intention_bonus = 0.0
    expected = section_attendue(intention)
    section = normaliser_recherche(meta.get("section", ""))

    if expected and expected.replace(".", " ") in section:
        intention_bonus += 0.45

    for mot in mots_cles_intention(intention):
        if normaliser_recherche(mot) in texte:
            intention_bonus += 0.06

    return min(base + intention_bonus, 1.0)


def respecte_filtre_medicament(chunk: dict[str, Any], filtre: str) -> bool:
    if not filtre:
        return True
    meta = chunk.get("metadata", {})
    filtre_n = normaliser_recherche(filtre)
    nom_n = normaliser_recherche(meta.get("medicament", ""))
    substance_n = normaliser_recherche(meta.get("substance", ""))

    if filtre_n in nom_n:
        return True

    premiers = filtre_n.split()
    if premiers and premiers[0] in nom_n:
        return True

    if filtre_n in substance_n:
        return True

    return False


def respecte_filtre_substance(chunk: dict[str, Any], filtre: str) -> bool:
    if not filtre:
        return True
    meta = chunk.get("metadata", {})
    filtre_n = normaliser_recherche(filtre)
    substance_n = normaliser_recherche(meta.get("substance", ""))
    nom_n = normaliser_recherche(meta.get("medicament", ""))

    return filtre_n in substance_n or filtre_n in nom_n


def rechercher_faiss_brut(
    question: str,
    modele: SentenceTransformer,
    index: faiss.Index,
    chunks_avec_meta: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    question_recherche = enrichir_question_pour_recherche(question)

    question_vecteur = modele.encode(
        [question_recherche],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(question_vecteur, k)

    resultats: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = dict(chunks_avec_meta[int(idx)])
        item["_idx"] = int(idx)
        item["score_faiss"] = float(score)
        item["score"] = float(score)
        resultats.append(item)

    return resultats


def rechercher_hybride(
    question: str,
    modele: SentenceTransformer,
    index: faiss.Index,
    chunks_avec_meta: list[dict[str, Any]],
    k: int,
    mode_recherche: str,
    filtre_medicament: str = "",
    filtre_substance: str = "",
    strict_medicament: bool = False,
) -> list[dict[str, Any]]:
    intention = intention_question(question)

    # On récupère plus de résultats au départ, puis on rerank.
    candidats = rechercher_faiss_brut(
        question=question,
        modele=modele,
        index=index,
        chunks_avec_meta=chunks_avec_meta,
        k=min(max(k * 6, 30), len(chunks_avec_meta)),
    )

    # Complément mots-clés : on scanne tout si filtre strict ou recherche substance,
    # sinon on ajoute seulement les chunks ayant des mots importants.
    termes = [normaliser_recherche(t) for t in termes_importants(question)]
    extra: list[dict[str, Any]] = []

    besoin_scan = strict_medicament or bool(filtre_medicament) or bool(filtre_substance)

    if besoin_scan:
        iterable = enumerate(chunks_avec_meta)
    else:
        iterable = enumerate(chunks_avec_meta[: min(len(chunks_avec_meta), 5000)])

    deja = {c.get("_idx") for c in candidats}

    for idx, chunk in iterable:
        if idx in deja:
            continue

        meta = chunk.get("metadata", {})
        texte = normaliser_recherche(
            " ".join(
                [
                    str(meta.get("medicament", "")),
                    str(meta.get("substance", "")),
                    str(meta.get("section", "")),
                    str(chunk.get("contenu", ""))[:2000],
                ]
            )
        )

        if filtre_medicament and not respecte_filtre_medicament(chunk, filtre_medicament):
            continue
        if filtre_substance and not respecte_filtre_substance(chunk, filtre_substance):
            continue

        if termes and not any(t in texte for t in termes):
            if not besoin_scan:
                continue

        item = dict(chunk)
        item["_idx"] = idx
        item["score_faiss"] = 0.0
        item["score"] = 0.0
        extra.append(item)

        if len(extra) > k * 8 and not besoin_scan:
            break

    candidats.extend(extra)

    # Filtres stricts.
    filtres: list[dict[str, Any]] = []
    for c in candidats:
        if filtre_medicament:
            if strict_medicament and not respecte_filtre_medicament(c, filtre_medicament):
                continue
            if not strict_medicament and not respecte_filtre_medicament(c, filtre_medicament):
                # En mode non strict, on autorise mais avec pénalité plus bas.
                pass
        if filtre_substance and not respecte_filtre_substance(c, filtre_substance):
            continue
        filtres.append(c)

    # Déduplication.
    uniques: dict[int, dict[str, Any]] = {}
    for c in filtres:
        idx = int(c.get("_idx", -1))
        if idx not in uniques or c.get("score_faiss", 0.0) > uniques[idx].get("score_faiss", 0.0):
            uniques[idx] = c

    reranked: list[dict[str, Any]] = []
    for c in uniques.values():
        faiss_score = float(c.get("score_faiss", c.get("score", 0.0)))
        kw_score = score_mots_cles(question, c, intention)

        if mode_recherche == "FAISS uniquement":
            final = faiss_score
        elif mode_recherche == "Mots-clés uniquement":
            final = kw_score
        else:
            final = 0.68 * faiss_score + 0.32 * kw_score

        # Bonus filtre médicament si match demandé.
        if filtre_medicament and respecte_filtre_medicament(c, filtre_medicament):
            final += 0.15

        expected = section_attendue(intention)
        section_norm = normaliser_recherche(c.get("metadata", {}).get("section", ""))
        if expected and expected.replace(".", " ") in section_norm:
            final += 0.10

        c["score_mots_cles"] = float(kw_score)
        c["score"] = float(final)
        reranked.append(c)

    reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return reranked[:k]


def contexte_global(chunks: list[dict[str, Any]]) -> str:
    parts = []

    for c in chunks:
        meta = c.get("metadata", {})
        parts.append(
            " ".join(
                [
                    str(meta.get("medicament", "")),
                    str(meta.get("substance", "")),
                    str(meta.get("section", "")),
                    str(c.get("contenu", "")),
                ]
            )
        )

    return "\n".join(parts)


def doit_refuser_avant_llm(
    question: str,
    chunks: list[dict[str, Any]],
    seuil_confiance: float,
) -> tuple[bool, str]:
    if not chunks:
        return True, "Aucun chunk récupéré par FAISS."

    meilleur_score = max(c.get("score", 0.0) for c in chunks)

    if meilleur_score < seuil_confiance:
        return True, f"Score de confiance insuffisant ({meilleur_score:.3f})."

    termes = termes_importants(question)
    contexte = contexte_global(chunks)
    termes_non_trouves = [t for t in termes if not fuzzy_present(t, contexte)]

    if termes_non_trouves and meilleur_score < 0.45:
        return True, "Terme important absent des sources récupérées : " + ", ".join(termes_non_trouves[:5])

    return False, ""


# ============================================================
# Construction du contexte LLM
# ============================================================

def extraire_extrait_pertinent(contenu: str, question: str, max_len: int = 1800) -> str:
    contenu = str(contenu)
    contenu_norm = sans_accents(contenu)

    intention = intention_question(question)
    mots_cles = mots_cles_intention(intention)

    positions = []

    for mot in mots_cles:
        mot_norm = sans_accents(mot)
        pos = contenu_norm.find(mot_norm)

        if pos != -1:
            positions.append(pos)

    if positions:
        pos = min(positions)
        start = max(0, pos - 500)
        end = min(len(contenu), pos + max_len)
        return contenu[start:end].strip()

    for terme in termes_importants(question):
        pos = contenu_norm.find(sans_accents(terme))

        if pos != -1:
            start = max(0, pos - 500)
            end = min(len(contenu), pos + max_len)
            return contenu[start:end].strip()

    return contenu[:max_len].strip()


def construire_contexte(chunks: list[dict[str, Any]], question: str, max_chars_source: int) -> str:
    blocs: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        cis = str(meta.get("cis", "non renseigné"))
        url = lien_bdpm(cis)

        contenu_pertinent = extraire_extrait_pertinent(
            contenu=chunk.get("contenu", ""),
            question=question,
            max_len=max_chars_source,
        )

        blocs.append(
            f"[SOURCE {i}]\n"
            f"Médicament : {meta.get('medicament', 'inconnu')}\n"
            f"Code CIS : {cis}\n"
            f"Lien BDPM : {url}\n"
            f"Substance : {meta.get('substance', 'inconnue')}\n"
            f"Section : {meta.get('section', 'inconnue')}\n"
            f"Score hybride : {chunk.get('score', 0):.3f}\n"
            f"Score FAISS : {chunk.get('score_faiss', chunk.get('score', 0)):.3f}\n"
            f"Score mots-clés : {chunk.get('score_mots_cles', 0):.3f}\n"
            f"Contenu pertinent : {contenu_pertinent}"
        )

    return "\n\n".join(blocs)


def construire_prompt_systeme() -> str:
    return f"""
Tu es un assistant RAG d'information sur les médicaments.
Tu dois répondre uniquement à partir du CONTEXTE fourni.
Tu n'as pas le droit d'utiliser tes connaissances générales.

Règles obligatoires :
1. Si le médicament demandé n'apparaît dans aucune source, réponds que tu ne trouves pas l'information dans la base.
2. Si le médicament apparaît mais que l'information demandée n'est pas présente dans le contexte, dis que cette information précise n'est pas trouvée.
3. Si le contexte contient des effets indésirables, une posologie, une contre-indication ou une interaction, tu dois les résumer clairement.
4. N'invente jamais une substance active, une posologie, une interaction ou un effet indésirable.
5. Cite les sources utilisées avec : nom du médicament, section, code CIS et lien BDPM si disponible.
6. Termine toujours par la phrase exacte : {MENTION_OBLIGATOIRE}
7. Ajoute : En cas de doute, consultez votre médecin ou votre pharmacien.
8. Ne donne pas de diagnostic.
9. Si plusieurs médicaments sont comparés, réponds sous forme de tableau clair quand c'est pertinent.

Format attendu :
- Réponse claire en français.
- Utilise des puces si la réponse contient plusieurs effets ou précautions.
- Ajoute une section "Sources utilisées" à la fin.
""".strip()


def reponse_refus(raison: str = "") -> str:
    details = f"\n\nDétail technique : {raison}" if raison else ""

    return (
        "Je ne trouve pas cette information dans ma base de connaissances. "
        "Je préfère ne pas inventer de réponse."
        f"{details}\n\n"
        f"{MENTION_OBLIGATOIRE}\n"
        "En cas de doute, consultez votre médecin ou votre pharmacien."
    )


def generer_reponse(
    question: str,
    chunks: list[dict[str, Any]],
    client: Groq,
    model_name: str,
    seuil_confiance: float,
    max_chars_source: int,
    max_tokens: int,
) -> str:
    refus, raison = doit_refuser_avant_llm(
        question=question,
        chunks=chunks,
        seuil_confiance=seuil_confiance,
    )

    if refus:
        return reponse_refus(raison)

    prompt_user = f"""
CONTEXTE :
{construire_contexte(chunks, question, max_chars_source)}

QUESTION UTILISATEUR :
{question}

Réponds uniquement avec les informations présentes dans le CONTEXTE.
""".strip()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            {"role": "user", "content": prompt_user},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    reponse = completion.choices[0].message.content.strip()

    if MENTION_OBLIGATOIRE not in reponse:
        reponse += f"\n\n{MENTION_OBLIGATOIRE}"

    if "consultez votre médecin" not in reponse.lower() and "consultez votre medecin" not in sans_accents(reponse):
        reponse += "\nEn cas de doute, consultez votre médecin ou votre pharmacien."

    return reponse


# ============================================================
# Interface Streamlit
# ============================================================

st.set_page_config(
    page_title="Assistant RAG Médicaments",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

st.markdown(
    """
    <style>
    /* ============================================================
       STYLE FINAL STABLE — clair, propre, sans texte invisible
       ============================================================ */

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 30%),
            radial-gradient(circle at top right, rgba(220, 38, 38, 0.08), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
        color: #0f172a !important;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* Texte principal lisible */
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container p,
    .block-container li,
    .block-container label,
    .block-container span {
        color: #0f172a;
    }

    /* Header */
    .hero {
        padding: 2.2rem 2.2rem;
        border-radius: 1.6rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 52%, #7f1d1d 100%) !important;
        border: 1px solid rgba(255,255,255,0.14);
        margin-bottom: 1.4rem;
        box-shadow: 0 22px 48px rgba(15, 23, 42, 0.24);
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-bottom: 0.45rem;
        line-height: 1.08;
    }

    .hero-subtitle {
        color: #e5e7eb !important;
        font-size: 1.05rem;
        max-width: 900px;
        line-height: 1.6;
    }

    .badge-row {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .badge {
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        color: #f9fafb !important;
        border: 1px solid rgba(255,255,255,0.2);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .hero *, .badge * {
        color: #ffffff !important;
    }

    /* Encadré médical */
    .warning-box {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412 !important;
        margin-bottom: 1.2rem;
        font-weight: 650;
    }

    .warning-box * {
        color: #9a3412 !important;
    }

    /* Cartes */
    .feature-card {
        padding: 1.2rem;
        border-radius: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.30);
        background: rgba(255,255,255,0.88);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        min-height: 150px;
    }

    .feature-card, .feature-card * {
        color: #0f172a !important;
    }

    .feature-card p, .small-muted {
        color: #64748b !important;
    }

    .answer-box {
        padding: 1.4rem;
        border-radius: 1.2rem;
        border: 1px solid rgba(34, 197, 94, 0.25);
        background: rgba(236, 253, 245, 0.92);
        margin-top: 0.8rem;
        box-shadow: 0 12px 28px rgba(34, 197, 94, 0.08);
    }

    .answer-box, .answer-box * {
        color: #064e3b !important;
    }

    .source-card {
        padding: 1rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.28);
        background: rgba(255,255,255,0.88);
        margin-bottom: 0.7rem;
    }

    .source-card, .source-card * {
        color: #0f172a !important;
    }

    .footer-note {
        color: #64748b !important;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 2rem;
    }

    /* Métriques visibles */
    .block-container div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        padding: 1rem !important;
        border-radius: 1rem !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05) !important;
    }

    .block-container div[data-testid="stMetric"] *,
    .block-container div[data-testid="stMetricValue"] *,
    .block-container div[data-testid="stMetricLabel"] * {
        color: #0f172a !important;
    }

    .block-container div[data-testid="stMetricLabel"] * {
        color: #64748b !important;
        font-weight: 700 !important;
    }

    /* Boutons */
    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stLinkButton"] > a {
        border-radius: 0.9rem !important;
        font-weight: 800 !important;
        min-height: 3rem;
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 10px 22px rgba(220, 38, 38, 0.22);
    }

    div.stButton > button[kind="primary"] *,
    div.stButton > button[data-testid="baseButton-primary"] * {
        color: #ffffff !important;
    }

    /* Inputs et selectbox : texte lisible */
    textarea, input {
        border-radius: 1rem !important;
        color: #0f172a !important;
        background: #ffffff !important;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-radius: 0.9rem !important;
        color: #0f172a !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #0f172a !important;
    }

    /* Onglets visibles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding-left: 1rem;
        padding-right: 1rem;
        background: rgba(255,255,255,0.88) !important;
        border: 1px solid rgba(148, 163, 184, 0.28);
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #334155 !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #fee2e2 !important;
        border-color: #ef4444 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span {
        color: #dc2626 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        border-radius: 0.9rem !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 750 !important;
    }

    /* Textarea des extraits : style sombre volontaire et lisible */
    div[data-testid="stTextArea"] textarea {
        background: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        font-family: Consolas, Monaco, monospace !important;
        font-size: 0.9rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] .stCaptionContainer *,
    section[data-testid="stSidebar"] small {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    section[data-testid="stSidebar"] code {
        color: #22c55e !important;
        background: rgba(15, 23, 42, 0.55) !important;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 1.4rem;
        }
        .hero-title {
            font-size: 2rem;
        }
    }


    /* ============================================================
       CORRECTION SIDEBAR — aucun bloc blanc / aucun texte invisible
       ============================================================ */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #172033 45%, #0f172a 100%) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    section[data-testid="stSidebar"] small {
        color: #cbd5e1 !important;
    }

    /* Expander sidebar : fond sombre, pas gris clair */
    section[data-testid="stSidebar"] div[data-testid="stExpander"],
    section[data-testid="stSidebar"] div[data-testid="stExpander"] details,
    section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] {
        background: rgba(15, 23, 42, 0.72) !important;
        border-color: rgba(255, 255, 255, 0.14) !important;
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        background: rgba(17, 24, 39, 0.95) !important;
        border-radius: 0.85rem 0.85rem 0 0 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary *,
    section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] *,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] label,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] p,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] span {
        color: #f8fafc !important;
    }

    /* Inputs dans la sidebar : champ clair mais texte noir lisible */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.55) !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] *,
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        color: #0f172a !important;
        fill: #0f172a !important;
    }

    /* Sliders dans la sidebar */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] *,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] label,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] span {
        color: #f8fafc !important;
    }

    /* Métriques sidebar : carte sombre lisible */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.09) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 1rem !important;
        padding: 1rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stMetric"] *,
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] *,
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] code {
        color: #22c55e !important;
        background: rgba(2, 6, 23, 0.78) !important;
        border-radius: 0.35rem !important;
        padding: 0.12rem 0.32rem !important;
        white-space: normal !important;
        word-break: break-word !important;
    }

    section[data-testid="stSidebar"] .stAlert {
        border-radius: 0.75rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Chargement ressources
# ============================================================

try:
    index, chunks_avec_meta = charger_index_et_chunks()
    modele = charger_modele_embedding()
except Exception as e:
    st.error(str(e))
    st.stop()

cfg = charger_config()
catalogue = construire_catalogue(chunks_avec_meta)

# ============================================================
# États session
# ============================================================

if "question" not in st.session_state:
    st.session_state["question"] = ""

if "historique" not in st.session_state:
    st.session_state["historique"] = []

if "filtre_medicament" not in st.session_state:
    st.session_state["filtre_medicament"] = ""

if "filtre_substance" not in st.session_state:
    st.session_state["filtre_substance"] = ""

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🧭 Assistant RAG")
    st.caption("BDPM + FAISS + mots-clés + Groq")

    api_key = os.getenv("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
    model_name = os.getenv("GROQ_MODEL", "") or st.secrets.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    if api_key:
        st.success("Service IA connecté")
    else:
        st.error("Clé Groq manquante")

    st.divider()

    st.markdown("## 📦 Base de connaissances")
    if cfg:
        st.success("Base vectorielle chargée")
        st.metric("Documents indexés", cfg.get("nb_chunks", index.ntotal))
        st.caption("Source : BDPM — Base de Données Publique des Médicaments")

        with st.expander("Voir les informations techniques"):
            st.write(f"**Modèle embedding :** `{cfg.get('embedding_model', EMBEDDING_MODEL_NAME)}`")
            st.write(f"**Index FAISS :** `{index.ntotal}` vecteurs")
            st.write(f"**Modèle Groq :** `{model_name}`")
            st.write("**Corpus :** BDPM officielle")
    else:
        st.info("Configuration de l’index non trouvée.")
        st.metric("Documents indexés", index.ntotal)

    st.divider()

    st.markdown("## 🧪 Questions rapides")
    exemples = [
        "Quels sont les effets indésirables de l’amoxicilline ?",
        "Quelle est la posologie de l’amoxicilline ?",
        "Quelles sont les contre-indications de l’ibuprofène ?",
        "Quels sont les effets indésirables du Doliprane ?",
        "Puis-je prendre Doliprane et ibuprofène en même temps ?",
    ]

    for ex in exemples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex
            st.rerun()

    st.divider()

    with st.expander("⚙️ Paramètres avancés", expanded=False):
        mode_recherche = st.selectbox(
            "Mode de recherche",
            ["Hybride FAISS + mots-clés", "FAISS uniquement", "Mots-clés uniquement"],
            index=0,
            help="Le mode hybride combine similarité sémantique et présence de mots-clés médicaux.",
        )

        top_k = st.slider(
            "Nombre de sources",
            min_value=3,
            max_value=14,
            value=TOP_K_DEFAULT,
            step=1,
        )

        seuil_confiance = st.slider(
            "Seuil de confiance",
            min_value=0.15,
            max_value=0.80,
            value=SEUIL_CONFIANCE_DEFAULT,
            step=0.01,
        )

        max_chars_source = st.slider(
            "Taille max par source",
            min_value=600,
            max_value=2600,
            value=1400,
            step=100,
            help="Réduis cette valeur si Groq affiche une erreur de tokens.",
        )

        max_tokens = st.slider(
            "Longueur max réponse",
            min_value=300,
            max_value=1100,
            value=650,
            step=50,
        )

    st.divider()

    st.markdown("## 🛡️ Sécurité")
    st.caption(
        "Le système utilise uniquement les sources récupérées. "
        "Il refuse de répondre si le contexte est insuffisant."
    )


client = Groq(api_key=api_key) if api_key else None

# ============================================================
# Header principal
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💊 Assistant RAG Médicaments</div>
        <div class="hero-subtitle">
            Pose une question sur un médicament. L’assistant combine recherche sémantique FAISS,
            mots-clés médicaux, filtres stricts et génération contrôlée avec Groq.
        </div>
        <div class="badge-row">
            <span class="badge">BDPM officielle</span>
            <span class="badge">Recherche hybride</span>
            <span class="badge">Filtre médicament</span>
            <span class="badge">Recherche substance</span>
            <span class="badge">Comparaison</span>
            <span class="badge">Export PDF</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
    ⚠️ Cet assistant fournit uniquement une aide informative à partir des sources indexées.
    Il ne remplace jamais l’avis d’un médecin ou d’un pharmacien.
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Sources indexées", index.ntotal)
col_b.metric("Médicaments détectés", len(catalogue["noms"]))
col_c.metric("Substances détectées", len(catalogue["substances"]))
col_d.metric("Mode", "RAG hybride")

st.divider()

# ============================================================
# Onglets
# ============================================================

tab_accueil, tab_chat, tab_comparaison, tab_recherche, tab_historique = st.tabs(
    [
        "🏠 Accueil",
        "💬 Chatbot RAG",
        "⚖️ Comparaison",
        "🔬 Recherche",
        "🕘 Historique",
    ]
)

# ============================================================
# Page accueil
# ============================================================

with tab_accueil:
    st.markdown("## Présentation du projet")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="feature-card">
            <h4>🔎 Recherche hybride</h4>
            <p>Combine FAISS avec un score mots-clés pour mieux récupérer les bonnes sections médicales.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="feature-card">
            <h4>🧪 Filtrage strict</h4>
            <p>Permet de limiter la recherche à un médicament précis ou à une substance active.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="feature-card">
            <h4>📚 Sources vérifiables</h4>
            <p>Chaque réponse affiche les sources récupérées, les scores et les liens BDPM officiels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Pipeline utilisé")
    st.code(
        """BDPM ZIP/CSV/HTML
→ Nettoyage HTML avec BeautifulSoup
→ Extraction des sections RCP
→ Chunking
→ Embeddings sentence-transformers
→ Index FAISS
→ Recherche hybride FAISS + mots-clés
→ Contexte contrôlé
→ Réponse Groq avec sources""",
        language="text",
    )

    st.markdown("### Comment utiliser l’application ?")
    st.markdown(
        """
        1. Va dans l’onglet **Chatbot RAG**.
        2. Pose une question sur un médicament.
        3. Active le **filtre strict** si tu veux éviter les médicaments proches.
        4. Consulte les sources officielles récupérées.
        5. Télécharge la réponse en **PDF** si nécessaire.
        """
    )

# ============================================================
# Chatbot RAG
# ============================================================

with tab_chat:
    st.markdown("### 🔎 Interroger la base médicamenteuse")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        choix_medicament = st.selectbox(
            "Filtre par médicament",
            options=[""] + catalogue["noms"][:5000],
            index=0,
            help="Optionnel. Utile pour forcer la recherche sur un médicament précis.",
        )

    with col_f2:
        choix_substance = st.selectbox(
            "Recherche par substance active",
            options=[""] + catalogue["substances"][:5000],
            index=0,
            help="Optionnel. Exemples : amoxicilline, ibuprofène, paracétamol.",
        )

    strict_medicament = st.toggle(
        "Activer le filtre strict par médicament",
        value=False,
        help="Si activé, les sources doivent correspondre au médicament choisi.",
    )

    question = st.text_area(
        "Pose ta question",
        value=st.session_state["question"],
        placeholder="Exemple : Quels sont les effets indésirables de l’amoxicilline ?",
        height=130,
    )

    intention_detectee = intention_question(question) if question.strip() else "general"
    st.caption(f"Intention détectée : **{libelle_intention(intention_detectee)}**")

    col_btn1, col_btn2, col_btn3 = st.columns([1.4, 1.2, 4])
    with col_btn1:
        lancer = st.button("🔎 Interroger", type="primary", use_container_width=True)
    with col_btn2:
        effacer = st.button("🧹 Effacer", use_container_width=True)
    with col_btn3:
        st.write("")

    if effacer:
        st.session_state["question"] = ""
        st.rerun()

    if lancer:
        if not question.strip():
            st.warning("Écris une question avant d’interroger le RAG.")
            st.stop()

        if client is None:
            st.error("GROQ_API_KEY manquante. Ajoute la clé dans le fichier .env en local ou dans les Secrets Streamlit Cloud.")
            st.stop()

        st.session_state["question"] = question

        # Si aucun filtre choisi, tentative de détection automatique dans la question.
        filtre_medicament_final = choix_medicament
        if not filtre_medicament_final:
            candidats = extraire_noms_medicaments_depuis_question(question, catalogue["noms"])
            if candidats:
                filtre_medicament_final = candidats[0]

        with st.spinner("Recherche hybride des sources pertinentes..."):
            chunks = rechercher_hybride(
                question=question,
                modele=modele,
                index=index,
                chunks_avec_meta=chunks_avec_meta,
                k=top_k,
                mode_recherche=mode_recherche,
                filtre_medicament=filtre_medicament_final,
                filtre_substance=choix_substance,
                strict_medicament=strict_medicament,
            )

        if not chunks:
            st.error("Aucune source récupérée. Essaie de désactiver le filtre strict ou de reformuler le médicament.")
            st.stop()

        best_score = max(c.get("score", 0.0) for c in chunks)

        st.markdown("### 📊 Résultat de la recherche")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sources récupérées", len(chunks))
        col2.metric("Meilleur score", f"{best_score:.3f}")
        col3.metric("Intention", libelle_intention(intention_detectee))
        col4.metric("Mode", mode_recherche.replace("Hybride ", ""))

        if filtre_medicament_final:
            st.info(f"Filtre médicament utilisé : **{filtre_medicament_final}**")
        if choix_substance:
            st.info(f"Filtre substance utilisé : **{choix_substance}**")

        with st.spinner("Génération de la réponse contrôlée avec Groq..."):
            try:
                reponse = generer_reponse(
                    question=question,
                    chunks=chunks,
                    client=client,
                    model_name=model_name,
                    seuil_confiance=seuil_confiance,
                    max_chars_source=max_chars_source,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                st.error("Erreur pendant l’appel Groq.")
                st.exception(e)
                st.info(
                    "Essaie de diminuer le nombre de sources ou la taille maximale par source dans les paramètres avancés."
                )
                st.stop()

        st.markdown("### 🤖 Réponse de l’assistant")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(reponse)
        st.markdown("</div>", unsafe_allow_html=True)

        export_txt = formater_reponse_telechargement(question, reponse, chunks)
        export_pdf = generer_pdf_simple("Assistant RAG Médicaments", export_txt)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="📥 Télécharger en TXT",
                data=export_txt,
                file_name="reponse_rag_medicaments.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="📄 Télécharger en PDF",
                data=export_pdf,
                file_name="reponse_rag_medicaments.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.session_state["historique"].insert(
            0,
            {
                "question": question,
                "reponse": reponse,
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "score": best_score,
                "sources": [
                    {
                        "medicament": c.get("metadata", {}).get("medicament", ""),
                        "section": c.get("metadata", {}).get("section", ""),
                        "cis": c.get("metadata", {}).get("cis", ""),
                        "score": c.get("score", 0.0),
                    }
                    for c in chunks
                ],
            },
        )

        st.divider()
        st.markdown("### 📚 Sources officielles récupérées")
        st.caption("Les sources ci-dessous sont les passages récupérés avant la génération de la réponse.")

        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})

            medicament = nettoyer_affichage(meta.get("medicament", "Médicament inconnu"), 220)
            section = nettoyer_affichage(meta.get("section", "Section inconnue"), 130)
            cis = nettoyer_affichage(meta.get("cis", "CIS inconnu"), 40)
            substance = nettoyer_affichage(meta.get("substance", "Non renseignée"), 180)
            score = float(chunk.get("score", 0.0))
            score_faiss = float(chunk.get("score_faiss", score))
            score_kw = float(chunk.get("score_mots_cles", 0.0))
            url = lien_bdpm(cis)

            titre = f"{i}. {medicament} — {section} — score {score:.3f}"

            with st.expander(titre, expanded=i <= 3):
                col_s1, col_s2, col_s3 = st.columns([2, 1, 1])

                with col_s1:
                    st.markdown(f"**Médicament :** {medicament}")
                    st.markdown(f"**Section RCP :** {section}")
                    st.markdown(f"**Substance active :** {substance}")

                with col_s2:
                    st.markdown(f"**Code CIS :** `{cis}`")
                    st.markdown(f"**Score hybride :** `{score:.3f}`")
                    st.markdown(f"**Score FAISS :** `{score_faiss:.3f}`")
                    st.markdown(f"**Score mots-clés :** `{score_kw:.3f}`")

                with col_s3:
                    if url:
                        st.link_button("🔗 Fiche BDPM", url, use_container_width=True, key=f"fiche_bdpm_chat_{i}_{cis}")

                extrait = extraire_extrait_pertinent(
                    contenu=chunk.get("contenu", ""),
                    question=question,
                    max_len=1200,
                )

                st.markdown("**Extrait utilisé :**")
                st.text_area(
                    label=f"Extrait source {i}",
                    value=extrait,
                    height=180,
                    label_visibility="collapsed",
                    key=f"chat_extrait_source_{i}_{cis}",
                )

# ============================================================
# Tableau comparatif entre deux médicaments
# ============================================================

with tab_comparaison:
    st.markdown("### ⚖️ Comparer deux médicaments")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        med1 = st.selectbox("Médicament 1", options=[""] + catalogue["noms"][:5000], index=0, key="med1_compare")
    with col_m2:
        med2 = st.selectbox("Médicament 2", options=[""] + catalogue["noms"][:5000], index=0, key="med2_compare")

    angle = st.selectbox(
        "Élément à comparer",
        [
            "Effets indésirables",
            "Posologie",
            "Contre-indications",
            "Interactions",
            "Information générale",
        ],
    )

    intention_map = {
        "Effets indésirables": "effets indésirables",
        "Posologie": "posologie",
        "Contre-indications": "contre-indications",
        "Interactions": "interactions",
        "Information générale": "information générale",
    }

    if st.button("⚖️ Générer le tableau comparatif", type="primary", use_container_width=True):
        if not med1 or not med2:
            st.warning("Choisis deux médicaments.")
            st.stop()

        if client is None:
            st.error("GROQ_API_KEY manquante.")
            st.stop()

        q1 = f"{intention_map[angle]} de {med1}"
        q2 = f"{intention_map[angle]} de {med2}"

        with st.spinner("Recherche des sources pour les deux médicaments..."):
            chunks1 = rechercher_hybride(
                question=q1,
                modele=modele,
                index=index,
                chunks_avec_meta=chunks_avec_meta,
                k=5,
                mode_recherche=mode_recherche,
                filtre_medicament=med1,
                strict_medicament=True,
            )
            chunks2 = rechercher_hybride(
                question=q2,
                modele=modele,
                index=index,
                chunks_avec_meta=chunks_avec_meta,
                k=5,
                mode_recherche=mode_recherche,
                filtre_medicament=med2,
                strict_medicament=True,
            )

        question_compare = (
            f"Compare sous forme de tableau les informations suivantes : {angle.lower()} "
            f"entre {med1} et {med2}. Réponds uniquement à partir des sources."
        )

        chunks_compare = chunks1 + chunks2

        if not chunks_compare:
            st.error("Aucune source trouvée pour la comparaison.")
            st.stop()

        with st.spinner("Génération de la comparaison..."):
            try:
                reponse_compare = generer_reponse(
                    question=question_compare,
                    chunks=chunks_compare,
                    client=client,
                    model_name=model_name,
                    seuil_confiance=0.15,
                    max_chars_source=1200,
                    max_tokens=900,
                )
            except Exception as e:
                st.error("Erreur pendant l’appel Groq.")
                st.exception(e)
                st.stop()

        st.markdown("### Résultat comparatif")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(reponse_compare)
        st.markdown("</div>", unsafe_allow_html=True)

        export_txt = formater_reponse_telechargement(question_compare, reponse_compare, chunks_compare)
        export_pdf = generer_pdf_simple("Comparaison RAG Médicaments", export_txt)

        st.download_button(
            "📄 Télécharger la comparaison en PDF",
            data=export_pdf,
            file_name="comparaison_medicaments.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.markdown("### Sources utilisées")
        for i, c in enumerate(chunks_compare, start=1):
            meta = c.get("metadata", {})
            med = nettoyer_affichage(meta.get("medicament", ""), 180)
            sec = nettoyer_affichage(meta.get("section", ""), 100)
            cis = nettoyer_affichage(meta.get("cis", ""), 40)
            score = float(c.get("score", 0.0))
            url = lien_bdpm(cis)

            with st.expander(f"{i}. {med} — {sec} — score {score:.3f}", expanded=i <= 2):
                st.write(f"Code CIS : `{cis}`")
                if url:
                    st.link_button("🔗 Fiche BDPM", url, key=f"fiche_bdpm_compare_{i}_{cis}")
                st.text_area(
                    "Extrait",
                    extraire_extrait_pertinent(c.get("contenu", ""), question_compare, 900),
                    height=140,
                    label_visibility="collapsed",
                    key=f"comparaison_extrait_{i}_{cis}",
                )

# ============================================================
# Recherche par substance ou médicament sans LLM
# ============================================================

with tab_recherche:
    st.markdown("### 🔬 Explorer la base sans générer de réponse")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        recherche_medicament = st.text_input("Rechercher un médicament", placeholder="Exemple : amoxicilline")
    with col_r2:
        recherche_substance = st.text_input("Rechercher une substance active", placeholder="Exemple : paracétamol")

    recherche_section = st.selectbox(
        "Filtrer par section",
        ["Toutes", "4.1 Indications", "4.2 Posologie", "4.3 Contre-indications", "4.5 Interactions", "4.8 Effets indésirables"],
    )

    if st.button("🔍 Rechercher dans les sources", use_container_width=True):
        resultats = []
        med_n = normaliser_recherche(recherche_medicament)
        sub_n = normaliser_recherche(recherche_substance)
        section_n = normaliser_recherche(recherche_section if recherche_section != "Toutes" else "")

        for chunk in chunks_avec_meta:
            meta = chunk.get("metadata", {})
            nom_n = normaliser_recherche(meta.get("medicament", ""))
            substance_n = normaliser_recherche(meta.get("substance", ""))
            section_chunk_n = normaliser_recherche(meta.get("section", ""))

            if med_n and med_n not in nom_n:
                continue
            if sub_n and sub_n not in substance_n:
                continue
            if section_n and section_n.split()[0] not in section_chunk_n:
                continue

            resultats.append(chunk)

            if len(resultats) >= 50:
                break

        st.metric("Résultats trouvés", len(resultats))

        for i, chunk in enumerate(resultats, start=1):
            meta = chunk.get("metadata", {})
            medicament = nettoyer_affichage(meta.get("medicament", "Médicament inconnu"), 220)
            section = nettoyer_affichage(meta.get("section", "Section inconnue"), 130)
            cis = nettoyer_affichage(meta.get("cis", "CIS inconnu"), 40)
            substance = nettoyer_affichage(meta.get("substance", "Non renseignée"), 180)
            url = lien_bdpm(cis)

            with st.expander(f"{i}. {medicament} — {section}", expanded=i <= 5):
                st.markdown(f"**Substance :** {substance}")
                st.markdown(f"**Code CIS :** `{cis}`")
                if url:
                    st.link_button("🔗 Fiche officielle BDPM", url, key=f"fiche_bdpm_recherche_{i}_{cis}")
                st.text_area(
                    "Extrait",
                    nettoyer_affichage(chunk.get("contenu", ""), 2500),
                    height=160,
                    label_visibility="collapsed",
                    key=f"recherche_extrait_{i}_{cis}",
                )

# ============================================================
# Historique
# ============================================================

with tab_historique:
    st.markdown("### 🕘 Historique de conversation")

    if not st.session_state["historique"]:
        st.info("Aucun échange pour le moment.")
    else:
        if st.button("🧹 Vider l’historique", use_container_width=True):
            st.session_state["historique"] = []
            st.rerun()

        all_history_text = []
        for item in st.session_state["historique"]:
            all_history_text.append(f"{item['date']} — {item['question']}\n{item['reponse']}\n")

        st.download_button(
            "📥 Télécharger tout l’historique",
            data="\n\n".join(all_history_text),
            file_name="historique_rag_medicaments.txt",
            mime="text/plain",
            use_container_width=True,
        )

        for item in st.session_state["historique"][:20]:
            with st.expander(f"{item['date']} — {item['question']}"):
                st.markdown(f"**Score max :** `{item['score']:.3f}`")
                st.markdown(item["reponse"])

                sources = item.get("sources", [])
                if sources:
                    st.markdown("**Sources :**")
                    for s in sources[:5]:
                        st.caption(
                            f"- {s.get('medicament', '')} — {s.get('section', '')} — "
                            f"CIS {s.get('cis', '')} — score {float(s.get('score', 0.0)):.3f}"
                        )

# ============================================================
# Pied de page
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        Projet RAG Médicaments — BDPM + FAISS + Sentence Transformers + Groq.
        <br>
        Usage pédagogique uniquement. Ne remplace pas un avis médical.
    </div>
    """,
    unsafe_allow_html=True,
)
