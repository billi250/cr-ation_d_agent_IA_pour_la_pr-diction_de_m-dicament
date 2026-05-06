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

TOP_K_DEFAULT = 5
SEUIL_CONFIANCE_DEFAULT = 0.38

MENTION_OBLIGATOIRE = "Ces informations ne remplacent pas l'avis d'un professionnel de santé."

BDPM_BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr/medicament/{cis}/extrait"

STOPWORDS_MED = {
    "quelle", "quelles", "quels", "quel", "sont", "est", "les", "des", "une", "dans", "avec", "pour",
    "effet", "effets", "secondaire", "secondaires", "indesirable", "indesirables", "posologie", "dose",
    "prendre", "peut", "puis", "combien", "comment", "medicament", "medicaments", "traitement",
    "risque", "contre", "indication", "indications", "utilisation", "utiliser", "adulte", "enfant",
    "femme", "grossesse", "allaitement", "quoi", "cest", "c", "donne", "explique",
}


# ============================================================
# Fonctions texte
# ============================================================

def sans_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


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


# ============================================================
# Intention de la question
# ============================================================

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
# Chargement FAISS / modèle
# ============================================================

@st.cache_resource(show_spinner=False)
def charger_modele_embedding() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def charger_index_et_chunks() -> tuple[faiss.Index, list[dict[str, Any]]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Base vectorielle introuvable. Lance d'abord : "
            "python indexation.py --corpus data/medicaments_corpus_bdpm.json"
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


# ============================================================
# Recherche FAISS
# ============================================================

def rechercher(
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
        item["score"] = float(score)
        resultats.append(item)

    return resultats


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

def extraire_extrait_pertinent(
    contenu: str,
    question: str,
    max_len: int = 1800,
) -> str:

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


def construire_contexte(
    chunks: list[dict[str, Any]],
    question: str,
    max_chars_source: int,
) -> str:

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
            f"Score : {chunk.get('score', 0):.3f}\n"
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
)

load_dotenv()

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.8rem;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.8rem;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">💊 Assistant RAG Médicaments</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interface Streamlit basée sur BDPM, FAISS, sentence-transformers et Groq.</div>',
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


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = os.getenv("GROQ_API_KEY", "")
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    if api_key:
        st.success("Clé Groq chargée")
    else:
        st.error("GROQ_API_KEY manquante dans le fichier .env")

    top_k = st.slider(
        "Nombre de sources FAISS",
        min_value=3,
        max_value=12,
        value=TOP_K_DEFAULT,
        step=1,
        help="Plus cette valeur est grande, plus le contexte est riche, mais plus il consomme de tokens.",
    )

    seuil_confiance = st.slider(
        "Seuil de confiance",
        min_value=0.20,
        max_value=0.80,
        value=SEUIL_CONFIANCE_DEFAULT,
        step=0.01,
    )

    max_chars_source = st.slider(
        "Taille max par source",
        min_value=700,
        max_value=3000,
        value=1600,
        step=100,
        help="Réduis cette valeur si Groq affiche une erreur de tokens.",
    )

    max_tokens = st.slider(
        "Longueur max réponse",
        min_value=300,
        max_value=900,
        value=600,
        step=50,
    )

    st.divider()

    st.subheader("📦 Index FAISS")

    cfg = charger_config()

    if cfg:
        st.write(f"**Modèle embedding :** `{cfg.get('embedding_model', EMBEDDING_MODEL_NAME)}`")
        st.write(f"**Chunks indexés :** `{cfg.get('nb_chunks', 'inconnu')}`")
        st.write(f"**Corpus :** `{Path(str(cfg.get('corpus_path', ''))).name}`")
    else:
        st.info("Aucune configuration trouvée.")

    st.divider()

    st.subheader("🧪 Exemples")

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


# ============================================================
# Chargement ressources
# ============================================================

try:
    index, chunks_avec_meta = charger_index_et_chunks()
    modele = charger_modele_embedding()
except Exception as e:
    st.error(str(e))
    st.stop()

client = Groq(api_key=api_key) if api_key else None

if "question" not in st.session_state:
    st.session_state["question"] = ""


# ============================================================
# Zone question
# ============================================================

question = st.text_area(
    "Pose ta question",
    value=st.session_state["question"],
    placeholder="Exemple : Quels sont les effets indésirables de l’amoxicilline ?",
    height=120,
)

col_btn1, col_btn2 = st.columns([1, 4])

with col_btn1:
    lancer = st.button("🔎 Interroger", type="primary", use_container_width=True)

with col_btn2:
    effacer = st.button("🧹 Effacer", use_container_width=True)

if effacer:
    st.session_state["question"] = ""
    st.rerun()


# ============================================================
# Exécution RAG
# ============================================================

if lancer:
    if not question.strip():
        st.warning("Écris une question avant d’interroger le RAG.")
        st.stop()

    if client is None:
        st.error("GROQ_API_KEY manquante. Vérifie ton fichier .env.")
        st.stop()

    with st.spinner("Recherche des sources pertinentes avec FAISS..."):
        chunks = rechercher(
            question=question,
            modele=modele,
            index=index,
            chunks_avec_meta=chunks_avec_meta,
            k=top_k,
        )

    if not chunks:
        st.error("Aucune source récupérée par FAISS.")
        st.stop()

    best_score = max(c.get("score", 0.0) for c in chunks)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sources récupérées", len(chunks))
    col2.metric("Meilleur score FAISS", f"{best_score:.3f}")
    col3.metric("Chunks indexés", index.ntotal)

    st.divider()

    with st.spinner("Génération de la réponse avec Groq..."):
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
                "Essaie de diminuer le nombre de sources FAISS ou la taille max par source dans la barre latérale."
            )
            st.stop()

    st.subheader("🤖 Réponse")
    st.markdown(reponse)

    st.divider()

    st.subheader("📚 Sources récupérées par FAISS")

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})

        medicament = nettoyer_affichage(meta.get("medicament", "Médicament inconnu"), 220)
        section = nettoyer_affichage(meta.get("section", "Section inconnue"), 130)
        cis = nettoyer_affichage(meta.get("cis", "CIS inconnu"), 40)
        substance = nettoyer_affichage(meta.get("substance", "Non renseignée"), 180)
        score = float(chunk.get("score", 0.0))
        url = lien_bdpm(cis)

        titre = f"{i}. {medicament} — {section} — score {score:.3f}"

        with st.expander(titre, expanded=i <= 3):
            st.markdown(f"**Médicament :** {medicament}")
            st.markdown(f"**Section :** {section}")
            st.markdown(f"**Code CIS :** `{cis}`")
            st.markdown(f"**Substance active :** {substance}")
            st.markdown(f"**Score FAISS :** `{score:.3f}`")

            if url:
                st.markdown(f"[🔗 Ouvrir la fiche officielle BDPM]({url})")

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
            )