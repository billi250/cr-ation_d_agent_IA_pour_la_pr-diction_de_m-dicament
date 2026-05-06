

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = BASE_DIR / "data" / "medicaments_corpus.json"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "medicaments.index"
CHUNKS_PATH = STORAGE_DIR / "chunks_medicaments.json"
CONFIG_PATH = STORAGE_DIR / "index_config.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Chunks plus petits = recherche plus précise.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


SECTION_KEYWORDS = {
    "4.8 Effets indésirables": [
        "effets indésirables",
        "effet indésirable",
        "effets secondaires",
        "troubles gastro-intestinaux",
        "nausées",
        "vomissements",
        "diarrhée",
        "éruption",
        "urticaire",
        "hypersensibilité",
        "réaction allergique",
        "anaphylactique",
        "fréquent",
        "rare",
        "très rare",
    ],
    "4.2 Posologie et mode d'administration": [
        "posologie",
        "mode d'administration",
        "dose",
        "doses",
        "administration",
        "prendre",
        "traitement",
        "adulte",
        "enfant",
        "nourrisson",
        "par jour",
    ],
    "4.3 Contre-indications": [
        "contre-indications",
        "contre indications",
        "contre-indication",
        "ne doit jamais être utilisé",
        "ne pas utiliser",
        "hypersensibilité",
        "allergie",
        "antécédent",
    ],
    "4.4 Mises en garde et précautions d'emploi": [
        "mises en garde",
        "précautions d'emploi",
        "précautions",
        "surveillance",
        "attention",
        "insuffisance rénale",
        "insuffisance hépatique",
    ],
    "4.5 Interactions avec d'autres médicaments": [
        "interactions",
        "interaction",
        "association",
        "autres médicaments",
        "médicaments",
        "anticoagulants",
        "méthotrexate",
    ],
    "4.1 Indications thérapeutiques": [
        "indications thérapeutiques",
        "indication thérapeutique",
        "indiqué",
        "traitement de",
        "infection",
    ],
    "4.6 Fertilité, grossesse et allaitement": [
        "grossesse",
        "allaitement",
        "fertilité",
        "femme enceinte",
    ],
    "4.9 Surdosage": [
        "surdosage",
        "dose excessive",
        "intoxication",
    ],
}


def nettoyer_texte(texte: str) -> str:
    if texte is None:
        return ""

    texte = str(texte).replace("\xa0", " ")
    texte = unicodedata.normalize("NFKC", texte)
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


def sans_accents(texte: str) -> str:
    texte = unicodedata.normalize("NFD", str(texte).lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def chunker(texte: str, taille_max: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte long en chunks avec chevauchement.
    """
    texte = nettoyer_texte(texte)

    if not texte:
        return []

    if len(texte) <= taille_max:
        return [texte]

    chunks: list[str] = []
    debut = 0
    n = len(texte)

    while debut < n:
        fin = min(debut + taille_max, n)
        morceau = texte[debut:fin]

        if fin < n:
            candidats = [
                morceau.rfind("\n\n"),
                morceau.rfind(". "),
                morceau.rfind("; "),
                morceau.rfind(": "),
            ]
            meilleur = max(candidats)

            if meilleur > taille_max * 0.45:
                fin = debut + meilleur + 1
                morceau = texte[debut:fin]

        morceau = morceau.strip()

        if len(morceau) > 40:
            chunks.append(morceau)

        if fin >= n:
            break

        debut = max(0, fin - overlap)

    return chunks


def charger_corpus(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Le corpus doit être une liste JSON non vide.")

    return corpus


def extraire_fenetre_autour_mot_cle(texte: str, mot_cle: str, avant: int = 1200, apres: int = 4500) -> str:
    
    texte_norm = sans_accents(texte)
    mot_norm = sans_accents(mot_cle)

    pos = texte_norm.find(mot_norm)

    if pos == -1:
        return ""

    start = max(0, pos - avant)
    end = min(len(texte), pos + apres)

    extrait = texte[start:end].strip()
    return nettoyer_texte(extrait)


def extraire_sections_depuis_rcp_complet(texte: str) -> dict[str, str]:
   
    texte = nettoyer_texte(texte)

    if not texte:
        return {}

    sections: dict[str, str] = {}

    for section_name, keywords in SECTION_KEYWORDS.items():
        extraits: list[str] = []

        for kw in keywords:
            extrait = extraire_fenetre_autour_mot_cle(texte, kw)

            if extrait and extrait not in extraits:
                extraits.append(extrait)

            if len(extraits) >= 3:
                break

        if extraits:
            sections[section_name] = "\n\n".join(extraits)[:12000]

    return sections


def est_section_rcp_complet(section: str) -> bool:
    s = sans_accents(section)
    return "rcp complet" in s or s.strip() in {"rcp", "complet"}


def construire_documents(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
   
    documents: list[dict[str, Any]] = []

    for med in corpus:
        med_id = nettoyer_texte(med.get("id", "med_sans_id"))
        cis = nettoyer_texte(med.get("cis", ""))
        nom = nettoyer_texte(med.get("medicament", "Médicament inconnu"))
        substance = nettoyer_texte(med.get("substance", "Non renseignée"))
        source = nettoyer_texte(med.get("source", "Corpus local"))

        sections = med.get("sections", {})

        if not isinstance(sections, dict):
            continue

        for section, contenu in sections.items():
            section_clean = nettoyer_texte(section)
            contenu_clean = nettoyer_texte(contenu)

            if not contenu_clean:
                continue

            # Si c'est un RCP complet, on essaye de créer des sections utiles.
            if est_section_rcp_complet(section_clean):
                sous_sections = extraire_sections_depuis_rcp_complet(contenu_clean)

                if sous_sections:
                    for sous_section, sous_contenu in sous_sections.items():
                        texte = (
                            f"Nom du médicament : {nom}\n"
                            f"Code CIS : {cis}\n"
                            f"Substance active : {substance}\n"
                            f"Rubrique / section : {sous_section}\n\n"
                            f"{sous_contenu}"
                        )

                        documents.append(
                            {
                                "id": f"{med_id}_{sous_section.lower().replace(' ', '_')}",
                                "contenu": texte,
                                "metadata": {
                                    "medicament": nom,
                                    "substance": substance,
                                    "section": sous_section,
                                    "source": source,
                                    "document_id": med_id,
                                    "cis": cis,
                                },
                            }
                        )

                    # On garde aussi un petit document général, mais moins prioritaire.
                    resume_general = contenu_clean[:3000]
                    texte_general = (
                        f"Nom du médicament : {nom}\n"
                        f"Code CIS : {cis}\n"
                        f"Substance active : {substance}\n"
                        f"Rubrique / section : RCP complet résumé\n\n"
                        f"{resume_general}"
                    )

                    documents.append(
                        {
                            "id": f"{med_id}_rcp_resume",
                            "contenu": texte_general,
                            "metadata": {
                                "medicament": nom,
                                "substance": substance,
                                "section": "RCP complet résumé",
                                "source": source,
                                "document_id": med_id,
                                "cis": cis,
                            },
                        }
                    )

                    continue

            # Cas normal : section déjà propre.
            texte = (
                f"Nom du médicament : {nom}\n"
                f"Code CIS : {cis}\n"
                f"Substance active : {substance}\n"
                f"Rubrique / section : {section_clean}\n\n"
                f"{contenu_clean}"
            )

            documents.append(
                {
                    "id": f"{med_id}_{section_clean.lower().replace(' ', '_')}",
                    "contenu": texte,
                    "metadata": {
                        "medicament": nom,
                        "substance": substance,
                        "section": section_clean,
                        "source": source,
                        "document_id": med_id,
                        "cis": cis,
                    },
                }
            )

    return documents


def construire_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Découpe les documents en chunks FAISS.
    """
    chunks_avec_meta: list[dict[str, Any]] = []

    for doc in documents:
        chunks = chunker(doc["contenu"])

        for i, chunk in enumerate(chunks):
            meta = dict(doc["metadata"])
            meta["chunk_id"] = f"{doc['id']}_chunk_{i}"
            meta["chunk_index"] = i

            chunks_avec_meta.append(
                {
                    "contenu": chunk,
                    "metadata": meta,
                }
            )

    return chunks_avec_meta


def embedder_chunks(chunks: list[str], modele: SentenceTransformer) -> np.ndarray:
    vecteurs = modele.encode(
        chunks,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vecteurs, dtype=np.float32)


def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    if vecteurs.ndim != 2:
        raise ValueError("Les vecteurs doivent avoir la forme (n_chunks, dimension).")

    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)

    return index


def sauvegarder(index: faiss.Index, chunks_avec_meta: list[dict[str, Any]], corpus_path: Path) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)

    config = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "corpus_path": str(corpus_path),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "nb_chunks": len(chunks_avec_meta),
        "faiss_index": "IndexFlatIP avec embeddings normalisés = similarité cosinus",
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Chemin du corpus JSON à indexer.",
    )

    args = parser.parse_args()

    corpus_path = Path(args.corpus) if args.corpus else DEFAULT_CORPUS_PATH

    if not corpus_path.is_absolute():
        corpus_path = BASE_DIR / corpus_path

    print(f"Chargement du corpus : {corpus_path}")

    corpus = charger_corpus(corpus_path)

    print(f"Médicaments chargés : {len(corpus)}")

    documents = construire_documents(corpus)

    print(f"Documents/sections construits : {len(documents)}")

    chunks_avec_meta = construire_chunks(documents)

    print(f"Chunks créés : {len(chunks_avec_meta)}")

    if not chunks_avec_meta:
        raise RuntimeError("Aucun chunk créé. Vérifie le corpus JSON.")

    print(f"Chargement modèle embedding : {EMBEDDING_MODEL_NAME}")

    modele = SentenceTransformer(EMBEDDING_MODEL_NAME)

    textes = [c["contenu"] for c in chunks_avec_meta]

    vecteurs = embedder_chunks(textes, modele)

    print(f"Vecteurs : {vecteurs.shape}")

    index = creer_index_faiss(vecteurs)

    sauvegarder(index, chunks_avec_meta, corpus_path)

    print("\n Indexation terminée")
    print(f"Index FAISS : {INDEX_PATH}")
    print(f"Chunks/meta : {CHUNKS_PATH}")
    print(f"Nombre de vecteurs : {index.ntotal}")


if __name__ == "__main__":
    main()