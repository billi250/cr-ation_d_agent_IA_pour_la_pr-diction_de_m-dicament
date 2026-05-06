from __future__ import annotations

import json
import os
import re
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


def formater_reponse_telechargement(question: str, reponse: str, chunks: list[dict[str, Any]]) -> str:
    lignes = [
        "Assistant RAG Médicaments",
        "=" * 40,
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
        score = float(chunk.get("score", 0.0))
        url = lien_bdpm(cis)

        lignes.append(f"{i}. {medicament} — {section} — CIS {cis} — score {score:.3f}")
        if url:
            lignes.append(f"   Lien : {url}")

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

    if any(mot in q for mot in ["contre indication", "contre-indication", "interdit", "ne pas prendre"]):
        return "contre_indications"

    if any(mot in q for mot in ["interaction", "interactions", "associer", "melanger", "avec"]):
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
    initial_sidebar_state="expanded",
)

load_dotenv()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 2rem 2rem;
        border-radius: 1.4rem;
        background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #991b1b 100%);
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 45px rgba(0,0,0,0.22);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        color: white;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #e5e7eb;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.6;
    }

    .badge-row {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .badge {
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        color: #f9fafb;
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.82rem;
        font-weight: 600;
    }

    .warning-box {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }

    .glass-card {
        padding: 1.2rem;
        border-radius: 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }

    .source-card {
        padding: 1rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(15, 23, 42, 0.04);
        margin-bottom: 0.7rem;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .answer-box {
        padding: 1.4rem;
        border-radius: 1.2rem;
        border: 1px solid rgba(34, 197, 94, 0.25);
        background: rgba(34, 197, 94, 0.04);
        margin-top: 0.8rem;
    }

    .footer-note {
        color: #6b7280;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 1rem;
        border-radius: 1rem;
    }

    div.stButton > button {
        border-radius: 0.9rem;
        font-weight: 700;
        height: 3rem;
    }

    textarea {
        border-radius: 1rem !important;
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

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.caption("Assistant basé sur une base vectorielle FAISS construite depuis la BDPM.")

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

    with st.expander("⚙️ Paramètres avancés"):
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

    st.markdown("## 🛡️ Sécurité")
    st.caption(
        "Le système utilise uniquement les sources récupérées par FAISS. "
        "Il refuse de répondre si le contexte est insuffisant."
    )


client = Groq(api_key=api_key) if api_key else None

if "question" not in st.session_state:
    st.session_state["question"] = ""

if "historique" not in st.session_state:
    st.session_state["historique"] = []


# ============================================================
# Header principal
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💊 Assistant RAG Médicaments</div>
        <div class="hero-subtitle">
            Pose une question sur un médicament. L’assistant recherche les passages pertinents
            dans la base officielle BDPM, récupère les sources avec FAISS, puis génère une réponse
            contrôlée avec Groq.
        </div>
        <div class="badge-row">
            <span class="badge">BDPM officielle</span>
            <span class="badge">FAISS</span>
            <span class="badge">Sentence Transformers</span>
            <span class="badge">Groq LLM</span>
            <span class="badge">Anti-hallucination</span>
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

col_a, col_b, col_c = st.columns(3)
col_a.metric("Sources disponibles", index.ntotal)
col_b.metric("Base", "BDPM")
col_c.metric("Mode", "RAG contrôlé")

st.divider()


# ============================================================
# Zone question
# ============================================================

st.markdown("### 🔎 Interroger la base médicamenteuse")

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


# ============================================================
# Exécution RAG
# ============================================================

if lancer:
    if not question.strip():
        st.warning("Écris une question avant d’interroger le RAG.")
        st.stop()

    if client is None:
        st.error("GROQ_API_KEY manquante. Ajoute la clé dans le fichier .env en local ou dans les Secrets Streamlit Cloud.")
        st.stop()

    st.session_state["question"] = question

    with st.spinner("Recherche sémantique des sources pertinentes avec FAISS..."):
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

    st.markdown("### 📊 Résultat de la recherche")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources récupérées", len(chunks))
    col2.metric("Meilleur score", f"{best_score:.3f}")
    col3.metric("Intention", libelle_intention(intention_detectee))
    col4.metric("Chunks indexés", index.ntotal)

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
                "Essaie de diminuer le nombre de sources FAISS ou la taille maximale par source dans les paramètres avancés."
            )
            st.stop()

    st.markdown("### 🤖 Réponse de l’assistant")

    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown(reponse)
    st.markdown("</div>", unsafe_allow_html=True)

    export_txt = formater_reponse_telechargement(question, reponse, chunks)

    st.download_button(
        label="📥 Télécharger la réponse et les sources",
        data=export_txt,
        file_name="reponse_rag_medicaments.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.session_state["historique"].insert(
        0,
        {
            "question": question,
            "reponse": reponse,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "score": best_score,
        },
    )

    st.divider()

    st.markdown("### 📚 Sources officielles récupérées")

    st.caption(
        "Les sources ci-dessous sont les passages récupérés par FAISS avant la génération de la réponse."
    )

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
            col_s1, col_s2 = st.columns([2, 1])

            with col_s1:
                st.markdown(f"**Médicament :** {medicament}")
                st.markdown(f"**Section RCP :** {section}")
                st.markdown(f"**Substance active :** {substance}")

            with col_s2:
                st.markdown(f"**Code CIS :** `{cis}`")
                st.markdown(f"**Score FAISS :** `{score:.3f}`")

                if url:
                    st.link_button("🔗 Fiche officielle BDPM", url, use_container_width=True)

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


# ============================================================
# Historique
# ============================================================

if st.session_state["historique"]:
    st.divider()
    st.markdown("### 🕘 Historique de la session")

    for item in st.session_state["historique"][:5]:
        with st.expander(f"{item['date']} — {item['question']}"):
            st.markdown(f"**Score max FAISS :** `{item['score']:.3f}`")
            st.markdown(item["reponse"])


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