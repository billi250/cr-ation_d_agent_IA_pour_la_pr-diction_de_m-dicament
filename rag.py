
from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "medicaments.index"
CHUNKS_PATH = STORAGE_DIR / "chunks_medicaments.json"
CONFIG_PATH = STORAGE_DIR / "index_config.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 12
SEUIL_CONFIANCE = 0.38
MENTION_OBLIGATOIRE = "Ces informations ne remplacent pas l'avis d'un professionnel de santé."

STOPWORDS_MED = {
    "quelle", "quelles", "quels", "quel", "sont", "est", "les", "des", "une", "dans", "avec", "pour",
    "effet", "effets", "secondaire", "secondaires", "indesirable", "indesirables", "posologie", "dose", "prendre",
    "peut", "puis", "combien", "comment", "medicament", "medicaments", "traitement", "risque", "contre",
    "indication", "indications", "utilisation", "utiliser", "adulte", "enfant", "femme", "grossesse", "allaitement",
}


def sans_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def termes_importants(question: str) -> list[str]:
    q = sans_accents(question)
    tokens = re.findall(r"[a-z0-9]{4,}", q)
    return [t for t in tokens if t not in STOPWORDS_MED]

def intention_question(question: str) -> str:
    q = sans_accents(question)

    if any(mot in q for mot in ["effet", "effets", "indesirable", "indesirables", "secondaire", "secondaires"]):
        return "effets_indesirables"

    if any(mot in q for mot in ["posologie", "dose", "prendre", "administration", "combien"]):
        return "posologie"

    if any(mot in q for mot in ["contre indication", "contre-indication", "interdit", "ne pas prendre"]):
        return "contre_indications"

    if any(mot in q for mot in ["interaction", "interactions", "associer", "melanger", "avec"]):
        return "interactions"

    return "general"


def enrichir_question_pour_recherche(question: str) -> str:
    """
    Ajoute des mots-clés médicaux pour aider FAISS à récupérer la bonne section.
    """
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
        ]

    if intention == "interactions":
        return [
            "interactions",
            "interactions avec d'autres médicaments",
            "association",
            "médicaments",
        ]

    return []

def fuzzy_present(term: str, text: str) -> bool:
    """Vérifie si un terme est présent ou presque présent dans le contexte."""
    text_norm = sans_accents(text)
    term_norm = sans_accents(term)
    if term_norm in text_norm:
        return True
    words = set(re.findall(r"[a-z0-9]{4,}", text_norm))
    for w in words:
        if abs(len(w) - len(term_norm)) <= 3 and SequenceMatcher(None, term_norm, w).ratio() >= 0.84:
            return True
    return False


def charger_index() -> tuple[faiss.Index, list[dict[str, Any]]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("Base vectorielle introuvable. Lance d'abord : python indexation.py")
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)
    return index, chunks_avec_meta

def rechercher(
    question: str,
    modele: SentenceTransformer,
    index: faiss.Index,
    chunks_avec_meta: list[dict[str, Any]],
    k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Recherche les chunks les plus proches.
    On enrichit la question pour aider FAISS à récupérer la bonne section médicale.
    """

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
        item["score"] = float(score)
        resultats.append(item)

    return resultats

def contexte_global(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        meta = c.get("metadata", {})
        parts.append(" ".join([
            str(meta.get("medicament", "")),
            str(meta.get("substance", "")),
            str(meta.get("section", "")),
            str(c.get("contenu", "")),
        ]))
    return "\n".join(parts)


def doit_refuser_avant_llm(question: str, chunks: list[dict[str, Any]]) -> tuple[bool, str]:
    if not chunks:
        return True, "Aucun chunk récupéré par FAISS."

    meilleur_score = max(c.get("score", 0.0) for c in chunks)

    if meilleur_score < SEUIL_CONFIANCE:
        return True, f"Score de confiance insuffisant ({meilleur_score:.3f})."

    termes = termes_importants(question)
    contexte = contexte_global(chunks)
    termes_non_trouves = [t for t in termes if not fuzzy_present(t, contexte)]

    # Important :
    # On ne refuse que si le terme absent ressemble au nom du médicament demandé.
    # On ne refuse pas juste parce que "effets indésirables" n'apparaît pas exactement.
    if termes_non_trouves and meilleur_score < 0.48:
        return True, "Terme important absent des sources récupérées : " + ", ".join(termes_non_trouves[:5])

    return False, ""

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
5. Cite les sources utilisées avec : nom du médicament, section, code CIS.
6. Termine toujours par la phrase exacte : {MENTION_OBLIGATOIRE}
7. Ajoute : En cas de doute, consultez votre médecin ou votre pharmacien.
8. Ne donne pas de diagnostic.

Format attendu :
- Réponse claire en français.
- Utilise des puces si la réponse contient plusieurs effets ou précautions.
- Ajoute une section "Sources" à la fin.
""".strip()

def extraire_extrait_pertinent(contenu: str, question: str, max_len: int = 5000) -> str:
    """
    Au lieu d'envoyer seulement le début du chunk, on cherche la partie utile :
    effets indésirables, posologie, contre-indications, interactions...
    """

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
        start = max(0, pos - 1000)
        end = min(len(contenu), pos + max_len)
        return contenu[start:end].strip()

    # Si aucun mot-clé de section n'est trouvé, on cherche les mots importants de la question.
    for terme in termes_importants(question):
        pos = contenu_norm.find(sans_accents(terme))

        if pos != -1:
            start = max(0, pos - 1000)
            end = min(len(contenu), pos + max_len)
            return contenu[start:end].strip()

    return contenu[:max_len].strip()

def construire_contexte(chunks: list[dict[str, Any]], question: str) -> str:
    blocs: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})

        contenu_pertinent = extraire_extrait_pertinent(
            contenu=chunk.get("contenu", ""),
            question=question,
            max_len=5000,
        )

        blocs.append(
            f"[SOURCE {i}]\n"
            f"Médicament : {meta.get('medicament', 'inconnu')}\n"
            f"Code CIS : {meta.get('cis', 'non renseigné')}\n"
            f"Substance : {meta.get('substance', 'inconnue')}\n"
            f"Section : {meta.get('section', 'inconnue')}\n"
            f"Score : {chunk.get('score', 0):.3f}\n"
            f"Contenu pertinent : {contenu_pertinent}"
        )

    return "\n\n".join(blocs)
    
def reponse_refus(raison: str = "") -> str:
    details = f"\nDétail technique : {raison}" if raison else ""
    return (
        "Je ne trouve pas cette information dans ma base de connaissances. "
        "Je préfère ne pas inventer de réponse."
        f"{details}\n\n"
        f"{MENTION_OBLIGATOIRE}\n"
        "En cas de doute, consultez votre médecin ou votre pharmacien."
    )


def generer_reponse(question: str, chunks: list[dict[str, Any]], client: Groq, model_name: str) -> str:
    refus, raison = doit_refuser_avant_llm(question, chunks)
    if refus:
        return reponse_refus(raison)

    prompt_user = f"""
CONTEXTE :
{construire_contexte(chunks, question)}

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
        max_tokens=900,
    )
    reponse = completion.choices[0].message.content.strip()

    # Post-sécurité : phrase obligatoire.
    if MENTION_OBLIGATOIRE not in reponse:
        reponse += f"\n\n{MENTION_OBLIGATOIRE}"
    if "consultez votre médecin" not in reponse.lower() and "consultez votre medecin" not in sans_accents(reponse):
        reponse += "\nEn cas de doute, consultez votre médecin ou votre pharmacien."
    return reponse

def clean_for_display(text: Any, max_len: int = 120) -> str:
    """Nettoie un texte avant affichage dans le terminal."""
    if text is None:
        return "Non renseigné"

    text = str(text)

    # Supprimer les balises HTML éventuelles.
    text = re.sub(r"<[^>]+>", " ", text)

    # Corriger quelques caractères HTML.
    text = (
        text.replace("&nbsp;", " ")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )

    # Nettoyer les espaces.
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "Non renseigné"

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text

def afficher_sources(chunks: list[dict[str, Any]]) -> None:
    print("\n--- Sources récupérées par FAISS ---")

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})

        medicament = clean_for_display(meta.get("medicament", "Médicament non renseigné"), 90)
        section = clean_for_display(meta.get("section", "Section non renseignée"), 80)
        cis = clean_for_display(meta.get("cis", "CIS inconnu"), 30)
        score = chunk.get("score", 0)

        print(f"{i}. {medicament} — {section} — CIS {cis} | score={score:.3f}")

    print("-----------------------------------\n")

    
def afficher_config() -> None:
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            print(f"Corpus indexé : {cfg.get('corpus_path')}")
            print(f"Nombre de chunks : {cfg.get('nb_chunks')}")
        except Exception:
            pass


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquante. Crée un fichier .env à partir de .env.example.")

    print("Chargement de la base de connaissances...")
    index, chunks_avec_meta = charger_index()
    print(f"Index FAISS chargé : {index.ntotal} vecteurs")
    afficher_config()

    print(f"Chargement du modèle d'embedding : {EMBEDDING_MODEL_NAME}")
    modele = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = Groq(api_key=api_key)

    print("\nAssistant RAG Médicaments prêt.")
    print("Tape 'quit', 'exit' ou 'q' pour quitter.")
    print("Exemples :")
    print("- Quels sont les effets indésirables du Voltarène ?")
    print("- Quelle est la posologie de l'amoxicilline ?")
    print("- Quelles sont les contre-indications de l'ibuprofène ?\n")

    while True:
        question = input("Votre question : ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir !")
            break
        if not question:
            continue
        chunks = rechercher(question, modele, index, chunks_avec_meta, k=TOP_K)
        reponse = generer_reponse(question, chunks, client, model_name)
        print("\n=== Réponse ===")
        print(reponse)
        afficher_sources(chunks)


if __name__ == "__main__":
    main()
