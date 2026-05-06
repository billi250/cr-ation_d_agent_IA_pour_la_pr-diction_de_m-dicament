<<<<<<< HEAD
# TP RAG — Sujet 2 : Assistant Médicaments avec Python, FAISS et Groq

Ce projet implémente un système **RAG complet** pour répondre à des questions sur des médicaments à partir d'une base de connaissances locale.

Le TP demandé impose notamment : chargement d'une base, nettoyage, chunking, embeddings, base vectorielle FAISS persistante, réponse avec Groq, citation des sources, refus d'inventer et avertissement médical systématique.

> Important : ce projet n'utilise ni LangChain ni LlamaIndex.

---

## 1. Architecture du projet

```text
rag_medicaments_tp/
│
├── data/
│   └── medicaments_corpus.json        # Corpus de départ structuré par médicament et section
│
├── storage/
│   ├── medicaments.index              # Généré après indexation
│   └── chunks_medicaments.json        # Généré après indexation
│
├── indexation.py                      # Phase 1 : création de la base FAISS
├── rag.py                             # Phase 2 : question-réponse avec Groq
├── test_install.py                    # Test rapide de FAISS + embeddings
├── requirements.txt                   # Dépendances Python
├── .env.example                       # Exemple de variable d'environnement
├── .gitignore                         # Empêche de publier la clé API
└── compte_rendu.md                    # Compte-rendu court demandé
```

---

## 2. Sujet choisi

J'ai choisi le **Sujet B — Assistant Médicaments**.

L'objectif est de construire un assistant capable de répondre à des questions comme :

- Quels sont les effets secondaires du Doliprane ?
- Y a-t-il des précautions entre ibuprofène et aspirine ?
- Quelle est la posologie de l'amoxicilline ?

Le système doit rester prudent, citer ses sources et toujours rappeler :

> Ces informations ne remplacent pas l'avis d'un professionnel de santé.

---

## 3. Source des données

Le TP recommande d'utiliser les données de la **Base de Données Publique des Médicaments (BDPM)**, source officielle mise à disposition par les autorités françaises.

Pour que le projet fonctionne immédiatement en local, le dossier contient un fichier :

```text
data/medicaments_corpus.json
```

Ce corpus contient 17 médicaments courants, structurés par sections : indications, posologie, contre-indications, effets indésirables, interactions.

Pour un rendu plus rigoureux, on peut remplacer ce corpus par des extraits officiels du fichier `CIS_RCP.zip` téléchargé depuis la BDPM.

---

## 4. Réponses aux questions de réflexion du sujet B

### Q1. Quelle stratégie de chunking pour les notices longues ?

Les notices sont longues, denses et découpées en sections médicales. J'ai choisi une taille de chunk de **900 caractères** avec un overlap de **120 caractères**. Cette taille garde assez de contexte sans envoyer trop de texte au LLM.

### Q2. Exploiter la structure des notices ?

Oui. Le corpus est structuré par sections :

- Indications
- Posologie
- Contre-indications
- Effets indésirables
- Interactions

Chaque section devient un document de base avant chunking. Cela améliore la précision de la recherche.

### Q3. Comment distinguer effets secondaires, posologie, interactions ?

Chaque chunk possède des métadonnées :

```json
{
  "medicament": "Ibuprofène",
  "substance": "Ibuprofène",
  "section": "Effets indésirables",
  "source": "Corpus local"
}
```

Ces métadonnées sont transmises au LLM pour qu'il cite clairement la source : médicament + section.

### Q4. Question sur deux médicaments ?

Si l'utilisateur demande par exemple :

> Puis-je prendre Doliprane et ibuprofène en même temps ?

La recherche vectorielle récupère les chunks liés aux deux médicaments. Le LLM synthétise uniquement les informations retrouvées. Si les chunks ne contiennent pas assez d'informations, il doit répondre qu'il ne trouve pas l'information dans la base.

### Q5. Prompt système prudent ?

Le prompt système impose au LLM :

- ne pas donner de diagnostic ;
- ne pas inventer ;
- citer le médicament et la section ;
- refuser si l'information n'est pas dans le contexte ;
- rappeler systématiquement que l'assistant ne remplace pas un professionnel de santé.

---

## 5. Installation sur Windows

### Étape 1 — Ouvrir PowerShell dans le dossier du projet

Dézippe le projet, puis va dans le dossier :

```powershell
cd C:\Users\lenovo\Desktop\rag_medicaments_tp
```

### Étape 2 — Créer un environnement virtuel

```powershell
python -m venv venv
```

### Étape 3 — Activer l'environnement

```powershell
venv\Scripts\activate
```

Si PowerShell bloque l'activation, exécute :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Puis relance :

```powershell
venv\Scripts\activate
```

### Étape 4 — Installer les dépendances

```powershell
pip install -r requirements.txt
```

### Étape 5 — Créer le fichier `.env`

Copie le fichier `.env.example` et renomme-le en `.env` :

```powershell
copy .env.example .env
```

Ouvre `.env` et remplace :

```text
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxx
```

par ta vraie clé Groq.

Tu peux créer une clé sur la console Groq.

---

## 6. Tester l'installation

```powershell
python test_install.py
```

Résultat attendu :

```text
Embedding OK — dimension : 768
FAISS OK — 1 vecteur(s) indexé(s)
Installation OK.
```

---

## 7. Créer la base vectorielle FAISS

À lancer une seule fois :

```powershell
python indexation.py
```

Ce script va générer :

```text
storage/medicaments.index
storage/chunks_medicaments.json
```

C'est important pour le TP : l'index est sauvegardé et n'est pas recréé à chaque question.

---

## 8. Lancer l'assistant RAG

```powershell
python rag.py
```

Exemples de questions :

```text
Quels sont les effets secondaires de l'ibuprofène ?
```

```text
Puis-je prendre Doliprane et ibuprofène en même temps ?
```

```text
Quelles sont les contre-indications de l'aspirine ?
```

```text
Quelle est la posologie de l'amoxicilline adulte ?
```

Pour quitter :

```text
quit
```

---

## 9. Choix techniques

### Modèle d'embedding

J'utilise :

```text
paraphrase-multilingual-mpnet-base-v2
```

Ce modèle est adapté au français et produit des vecteurs de dimension 768.

### FAISS

J'utilise :

```python
faiss.IndexFlatIP
```

Les embeddings sont normalisés. Le produit scalaire correspond donc à une similarité cosinus. Plus le score est proche de 1, plus le chunk est pertinent.

### Groq

Le PDF du TP mentionne `llama3-8b-8192`, mais ce modèle est maintenant déprécié. J'utilise donc :

```text
llama-3.1-8b-instant
```

Le modèle peut être changé dans `.env` :

```text
GROQ_MODEL=llama-3.1-8b-instant
```

---

## 10. Bonus implémenté

J'ai implémenté le bonus **score de confiance** :

- chaque chunk récupéré affiche un score FAISS ;
- si le meilleur score est inférieur à `0.30`, le système refuse de répondre ;
- cela limite les hallucinations.

---

## 11. Limites du projet

- Le corpus fourni est volontairement limité à 17 médicaments pour garder une indexation rapide.
- Le système ne remplace pas une vraie base médicale complète.
- Les réponses dépendent de la qualité du corpus et du chunking.
- Pour un rendu final plus solide, il est préférable de remplacer le corpus par des extraits officiels du fichier BDPM `CIS_RCP.zip`.

---

## 12. Commandes résumé

```powershell
cd C:\Users\lenovo\Desktop\rag_medicaments_tp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python test_install.py
python indexation.py
python rag.py
```
=======
# cr-ation_d_agent_IA_pour_la_pr-diction_de_m-dicament
>>>>>>> 912eac4653359e1f6adfdc31a691168e1fb0ae69
