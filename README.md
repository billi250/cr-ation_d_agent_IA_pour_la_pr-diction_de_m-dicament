# TP RAG — Sujet 2 : Assistant Médicaments avec BDPM, FAISS, Groq et Streamlit

Ce projet implémente un système **RAG complet** (*Retrieval-Augmented Generation*) permettant de répondre à des questions sur les médicaments à partir d’une base de connaissances construite avec des données issues de la **Base de Données Publique des Médicaments (BDPM)**.

Le projet répond au **Sujet 2 : Assistant Médicaments** du TP.

L’objectif est de construire un assistant capable de :

- charger une base documentaire réelle ;
- nettoyer des fichiers contenant du texte HTML ;
- structurer les notices de médicaments ;
- découper les documents en chunks ;
- calculer des embeddings ;
- créer une base vectorielle avec FAISS ;
- récupérer uniquement les passages pertinents ;
- interroger un LLM via Groq ;
- citer les sources utilisées ;
- éviter les hallucinations ;
- proposer une interface utilisateur avec Streamlit.

> Important : ce projet n’utilise ni LangChain ni LlamaIndex.  
> Le pipeline RAG est implémenté manuellement en Python.

---

## 1. Sujet choisi

J’ai choisi le **Sujet 2 : Assistant Médicaments**.

Le but est de créer un chatbot capable de répondre à des questions comme :

```text
Quels sont les effets indésirables de l’amoxicilline ?
```

```text
Quelle est la posologie de l’amoxicilline ?
```

```text
Quelles sont les contre-indications de l’ibuprofène ?
```

```text
Quels sont les effets secondaires du Doliprane ?
```

```text
Puis-je prendre deux médicaments ensemble ?
```

L’assistant ne doit pas répondre à partir de ses connaissances générales.  
Il doit répondre uniquement à partir des sources récupérées dans la base vectorielle.

---

## 2. Lien de l’application Streamlit

L’application est déployée en ligne avec **Streamlit Community Cloud**.

Lien de l’application :

```text
https://cr-ationdagentiapourlapr-dictiondem-dicament-mqbcfubdiw83s7d2t.streamlit.app/
```

Cette interface permet d’utiliser le RAG comme un vrai chatbot, sans passer par le terminal.

---

## 3. Avertissement médical

Ce projet est un projet pédagogique.

Les réponses générées ne remplacent jamais l’avis d’un médecin, d’un pharmacien ou d’un professionnel de santé.

L’assistant rappelle systématiquement :

```text
Ces informations ne remplacent pas l'avis d'un professionnel de santé.
En cas de doute, consultez votre médecin ou votre pharmacien.
```

Le chatbot ne donne pas de diagnostic médical et ne doit pas être utilisé pour prendre une décision de santé.

---

## 4. Fonctionnement général du RAG

Le fonctionnement du projet suit le pipeline RAG suivant :

```text
Fichiers BDPM
    ↓
Lecture des fichiers ZIP, CSV et TXT
    ↓
Détection de l’encodage et du séparateur
    ↓
Nettoyage du HTML avec BeautifulSoup
    ↓
Extraction des sections utiles du RCP
    ↓
Création d’un corpus JSON structuré
    ↓
Découpage du corpus en chunks
    ↓
Calcul des embeddings avec sentence-transformers
    ↓
Création d’un index FAISS
    ↓
Recherche des chunks proches de la question
    ↓
Construction du contexte
    ↓
Envoi du contexte au modèle Groq
    ↓
Génération d’une réponse sourcée
```

Le principe important est que le LLM ne reçoit jamais toute la base de données.  
Il reçoit uniquement les extraits sélectionnés par FAISS comme étant les plus proches de la question.

Cela permet :

- de réduire la consommation de tokens ;
- d’éviter de saturer le modèle ;
- de limiter les hallucinations ;
- d’obtenir des réponses justifiées par des sources.

---

## 5. Source des données

Les données utilisées proviennent de la **Base de Données Publique des Médicaments (BDPM)**.

Les fichiers utilisés dans ce projet sont principalement :

```text
CIS_RCP.zip
CIS_bdpm_officielle.zip
CIS_COMPO_bdpm.txt
```

Ces fichiers contiennent :

- le code CIS des médicaments ;
- la dénomination des médicaments ;
- la composition ;
- les substances actives ;
- les résumés des caractéristiques du produit ;
- les indications ;
- la posologie ;
- les contre-indications ;
- les interactions ;
- les effets indésirables ;
- les informations de surdosage.

Contrairement à une première version basée sur un petit corpus local, cette version utilise une base réelle issue de la BDPM.

---

## 6. Problème rencontré avec les fichiers BDPM

Les fichiers BDPM ne sont pas toujours simples à lire directement.

Les principaux problèmes rencontrés sont :

- fichiers compressés en `.zip` ;
- fichiers `.csv` ou `.txt` volumineux ;
- encodage parfois en `latin-1` ;
- encodage parfois en `utf-8-sig` ;
- séparateur pouvant être une tabulation ;
- contenu médical stocké avec des balises HTML ;
- très grand nombre de lignes dans `CIS_RCP.csv`.

Le fichier RCP peut contenir plusieurs millions de lignes.  
Il ne faut donc pas le lire naïvement en mémoire sans optimisation.

---

## 7. Solution utilisée pour lire les fichiers

Le script `importer_bdpm.py` gère automatiquement plusieurs cas.

### 7.1 Décompression des fichiers ZIP

Les fichiers `.zip` sont automatiquement décompressés dans :

```text
data/bdpm_raw/
```

Exemple :

```text
data/CIS_RCP.zip
```

est décompressé dans :

```text
data/bdpm_raw/CIS_RCP/
```

Cela évite à l’utilisateur de décompresser manuellement les fichiers.

### 7.2 Détection de l’encodage

Le script teste plusieurs encodages :

```text
utf-8-sig
utf-8
latin-1
```

Cela permet d’éviter les problèmes de caractères du type `Ã©`, `Ã¨`, `Ãª`, qui apparaissent souvent quand un fichier est lu avec le mauvais encodage.

### 7.3 Détection du séparateur

Le script teste plusieurs séparateurs :

```text
tabulation
point-virgule
virgule
```

Le meilleur séparateur est choisi en fonction de la structure du tableau obtenu.

### 7.4 Lecture par blocs

Le fichier `CIS_RCP.csv` est très grand.

Pour éviter de bloquer le PC, il est lu par blocs avec `chunksize`.

Cela permet de traiter progressivement les lignes sans charger tout le fichier en mémoire.

### 7.5 Nettoyage du HTML

Le contenu des RCP contient souvent du HTML.

Le projet utilise **BeautifulSoup** pour transformer ce contenu en texte propre.

Le nettoyage consiste à :

- supprimer les balises HTML ;
- supprimer les balises `script` et `style` ;
- récupérer le texte lisible ;
- nettoyer les espaces ;
- normaliser les caractères ;
- conserver les sections médicales importantes.

---

## 8. Sections médicales extraites

Le script extrait les sections importantes du RCP.

Exemples :

```text
1 Dénomination du médicament
2 Composition qualitative et quantitative
3 Forme pharmaceutique
4.1 Indications thérapeutiques
4.2 Posologie et mode d'administration
4.3 Contre-indications
4.4 Mises en garde et précautions d'emploi
4.5 Interactions avec d'autres médicaments
4.6 Fertilité, grossesse et allaitement
4.8 Effets indésirables
4.9 Surdosage
5.1 Propriétés pharmacodynamiques
5.2 Propriétés pharmacocinétiques
6 Données pharmaceutiques
```

Cette structuration est importante, car elle permet au RAG de retrouver plus précisément les informations demandées.

Par exemple :

- une question sur la posologie doit récupérer plutôt la section `4.2` ;
- une question sur les effets indésirables doit récupérer plutôt la section `4.8` ;
- une question sur les contre-indications doit récupérer plutôt la section `4.3` ;
- une question sur les interactions doit récupérer plutôt la section `4.5`.

---

## 9. Architecture du projet

```text
rag_medicaments_tp/
│
├── data/
│   ├── CIS_RCP.zip
│   ├── CIS_bdpm_officielle.zip
│   ├── CIS_COMPO_bdpm.txt
│   ├── bdpm_raw/
│   └── medicaments_corpus_bdpm.json
│
├── storage/
│   ├── medicaments.index
│   ├── chunks_medicaments.json
│   └── index_config.json
│
├── importer_bdpm.py
├── indexation.py
├── rag.py
├── app_streamlit.py
├── test_install.py
├── requirements.txt
├── compte_rendu.md
├── .env.example
├── .gitignore
└── README.md
```

---

## 10. Rôle des fichiers principaux

### `importer_bdpm.py`

Ce fichier sert à importer les données BDPM.

Il réalise les étapes suivantes :

- décompresser les fichiers ZIP ;
- lire les fichiers BDPM ;
- détecter l’encodage ;
- détecter le séparateur ;
- nettoyer le HTML ;
- extraire les sections des RCP ;
- regrouper les informations par médicament ;
- générer un corpus JSON.

Fichier généré :

```text
data/medicaments_corpus_bdpm.json
```

### `indexation.py`

Ce fichier crée la base vectorielle.

Il réalise les étapes suivantes :

- charger le corpus JSON ;
- transformer les sections en documents ;
- découper les documents en chunks ;
- calculer les embeddings ;
- normaliser les vecteurs ;
- créer un index FAISS ;
- sauvegarder l’index et les chunks.

Fichiers générés :

```text
storage/medicaments.index
storage/chunks_medicaments.json
storage/index_config.json
```

### `rag.py`

Ce fichier permet d’utiliser le chatbot dans le terminal.

Il réalise les étapes suivantes :

- charger l’index FAISS ;
- charger les chunks ;
- charger le modèle d’embedding ;
- transformer la question en vecteur ;
- rechercher les chunks les plus proches ;
- construire le contexte ;
- appeler Groq ;
- afficher la réponse ;
- afficher les sources récupérées.

### `app_streamlit.py`

Ce fichier contient l’interface web.

Il permet :

- de poser une question dans une interface graphique ;
- de récupérer les sources FAISS ;
- d’afficher la réponse du chatbot ;
- d’afficher les sources ;
- d’afficher le score FAISS ;
- d’ouvrir les fiches officielles BDPM ;
- de consulter les extraits utilisés.

### `requirements.txt`

Ce fichier contient les dépendances nécessaires au projet.

Exemple :

```text
streamlit
pandas
numpy
faiss-cpu
sentence-transformers
groq
python-dotenv
beautifulsoup4
tqdm
```

Important : il ne faut pas écrire de commandes comme `pip install streamlit` dans `requirements.txt`.

---

## 11. Installation locale sur Windows

### Étape 1 — Cloner le dépôt

```powershell
git clone https://github.com/billi250/cr-ation_d_agent_IA_pour_la_pr-diction_de_m-dicament.git
cd cr-ation_d_agent_IA_pour_la_pr-diction_de_m-dicament
```

### Étape 2 — Créer un environnement virtuel

```powershell
python -m venv venv
```

### Étape 3 — Activer l’environnement virtuel

```powershell
venv\Scripts\activate
```

Si PowerShell bloque l’activation :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Puis relancer :

```powershell
venv\Scripts\activate
```

### Étape 4 — Installer les dépendances

```powershell
pip install -r requirements.txt
```

### Étape 5 — Créer le fichier `.env`

Créer un fichier `.env` à la racine du projet :

```text
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.1-8b-instant
```

La clé API Groq ne doit jamais être publiée sur GitHub.

---

## 12. Tester l’installation

Lancer :

```powershell
python test_install.py
```

Résultat attendu :

```text
Embedding OK
FAISS OK
Installation OK
```

---

## 13. Préparer les données BDPM

Placer les fichiers BDPM dans le dossier `data/`.

Exemple :

```text
data/CIS_RCP.zip
data/CIS_bdpm_officielle.zip
data/CIS_COMPO_bdpm.txt
```

Ensuite lancer l’importation.

Pour importer 1000 médicaments :

```powershell
python importer_bdpm.py --limit 1000
```

Pour importer 10000 médicaments :

```powershell
python importer_bdpm.py --limit 10000
```

Pour importer tous les médicaments :

```powershell
python importer_bdpm.py --limit 0
```

Pour filtrer sur un médicament précis :

```powershell
python importer_bdpm.py --query amoxicilline --limit 20
```

Le fichier généré est :

```text
data/medicaments_corpus_bdpm.json
```

---

## 14. Créer l’index FAISS

Après avoir généré le corpus JSON :

```powershell
python indexation.py --corpus data/medicaments_corpus_bdpm.json
```

Cette étape crée les embeddings et l’index FAISS.

Fichiers générés :

```text
storage/medicaments.index
storage/chunks_medicaments.json
storage/index_config.json
```

---

## 15. Lancer le RAG en terminal

```powershell
python rag.py
```

Exemples de questions :

```text
Quels sont les effets indésirables de l’amoxicilline ?
```

```text
Quelle est la posologie de l’amoxicilline ?
```

```text
Quelles sont les contre-indications de l’ibuprofène ?
```

```text
C’est quoi le Doliprane ?
```

Pour quitter : `quit` ou `exit`.

---

## 16. Lancer l’interface Streamlit en local

```powershell
streamlit run app_streamlit.py
```

L’application s’ouvre dans le navigateur :

```text
http://localhost:8501
```

---

## 17. Déploiement en ligne avec Streamlit Cloud

Le projet peut être déployé avec **Streamlit Community Cloud**.

Étapes générales :

1. publier le projet sur GitHub ;
2. aller sur Streamlit Community Cloud ;
3. créer une nouvelle application ;
4. choisir le dépôt GitHub ;
5. choisir la branche `main` ;
6. choisir le fichier principal : `app_streamlit.py` ;
7. ajouter la clé API dans les secrets Streamlit.

---

## 18. Gestion de la clé API en déploiement

Le fichier `.env` n’est pas envoyé sur GitHub pour des raisons de sécurité.

En local, la clé est lue depuis `.env`.

En ligne, la clé est lue depuis les **Secrets** de Streamlit Cloud.

Exemple de configuration dans Streamlit Secrets :

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

## 19. Explication des embeddings

Chaque chunk de texte est transformé en vecteur numérique grâce au modèle :

```text
paraphrase-multilingual-mpnet-base-v2
```

Ce vecteur représente le sens du texte.

La question de l’utilisateur est aussi transformée en vecteur.

FAISS compare ensuite le vecteur de la question avec les vecteurs des chunks.

Les chunks les plus proches sont récupérés et envoyés au LLM.

Cela évite d’envoyer toute la base de données au modèle Groq.

---

## 20. Modèle d’embedding utilisé

Le modèle utilisé est :

```text
paraphrase-multilingual-mpnet-base-v2
```

Il est adapté aux textes en français et produit des vecteurs de dimension 768.

---

## 21. Base vectorielle FAISS

Le projet utilise FAISS pour stocker et rechercher les embeddings.

Les embeddings sont normalisés.

L’index FAISS permet donc de comparer les textes avec une similarité proche de la similarité cosinus.

L’index est sauvegardé dans :

```text
storage/medicaments.index
```

Cela évite de recalculer les embeddings à chaque lancement du chatbot.

---

## 22. Modèle Groq utilisé

Le modèle Groq utilisé par défaut est :

```text
llama-3.1-8b-instant
```

Le modèle est appelé uniquement après la récupération des sources par FAISS.

Le LLM reçoit :

- la question utilisateur ;
- les extraits pertinents ;
- les métadonnées des sources ;
- les consignes anti-hallucination.

---

## 23. Gestion anti-hallucination

Le projet contient plusieurs mécanismes pour limiter les hallucinations.

### 23.1 Réponse basée uniquement sur le contexte

Le prompt système interdit au modèle d’utiliser ses connaissances générales.

### 23.2 Score FAISS

Chaque source récupérée possède un score de similarité.

Si le score de similarité est trop faible, l’assistant refuse de répondre au lieu d’inventer.

### 23.3 Vérification des termes importants

Le système extrait les mots importants de la question.

Si un médicament ou un terme important n’apparaît pas dans les sources récupérées, le système peut refuser de répondre.

### 23.4 Réduction du contexte envoyé au LLM

Pour éviter les erreurs de limite de tokens avec Groq, l’application n’envoie pas tout le contenu des sources au LLM.

Elle extrait uniquement les passages les plus pertinents autour des mots-clés de la question.

### 23.5 Citation des sources

Les réponses doivent citer les sources utilisées, avec :

- le nom du médicament ;
- la section ;
- le code CIS ;
- le lien officiel BDPM quand il est disponible.

---

## 24. Recherche orientée par intention

Le système détecte l’intention de la question.

### Effets indésirables

Si la question contient des termes comme `effets indésirables`, `effets secondaires`, `nausées`, `réactions allergiques`, la recherche favorise la section :

```text
4.8 Effets indésirables
```

### Posologie

Si la question contient des termes comme `posologie`, `dose`, `prendre`, `mode d’administration`, la recherche favorise la section :

```text
4.2 Posologie et mode d'administration
```

### Contre-indications

Si la question contient des termes comme `contre-indication`, `allergie`, `ne pas prendre`, `interdit`, la recherche favorise la section :

```text
4.3 Contre-indications
```

### Interactions

Si la question contient des termes comme `interaction`, `mélanger`, `associer`, `avec un autre médicament`, la recherche favorise la section :

```text
4.5 Interactions avec d'autres médicaments
```

---

## 25. Interface Streamlit

L’interface Streamlit contient :

- un champ de question ;
- un bouton d’interrogation ;
- des exemples de questions ;
- un affichage de la réponse ;
- les sources récupérées ;
- les scores FAISS ;
- les sections utilisées ;
- les codes CIS ;
- les liens vers les fiches officielles BDPM ;
- les extraits utilisés.

L’objectif est de rendre le projet plus professionnel et plus simple à tester par l’enseignant.

---

## 26. Exemple de réponse

Question :

```text
Quels sont les effets indésirables de l’amoxicilline ?
```

Exemple de réponse attendue :

```text
Les effets indésirables mentionnés dans les sources récupérées peuvent inclure :

- nausées ;
- vomissements ;
- diarrhée ;
- réactions cutanées ;
- manifestations allergiques ;
- troubles digestifs.

Ces informations ne remplacent pas l'avis d'un professionnel de santé.
En cas de doute, consultez votre médecin ou votre pharmacien.
```

Les sources affichées permettent ensuite de vérifier les informations :

```text
Médicament : AMOXICILLINE ...
Section : 4.8 Effets indésirables
Code CIS : ...
Score FAISS : ...
```

---

## 27. Réponses aux questions de réflexion du sujet

### Q1. Quelle stratégie de chunking pour les notices longues ?

Les notices sont longues et contiennent plusieurs sections médicales.

La stratégie utilisée consiste à extraire les sections importantes du RCP, puis à indexer ces sections sous forme de documents ou de chunks.

Cela permet de ne pas envoyer toute une notice au LLM.

### Q2. Comment exploiter la structure des notices ?

Les notices/RCP contiennent des sections officielles.

Le projet conserve ces sections dans les métadonnées :

```json
{
  "medicament": "AMOXICILLINE ...",
  "cis": "66082724",
  "section": "4.8 Effets indésirables",
  "source": "BDPM - CIS_RCP.zip"
}
```

Ces métadonnées sont utilisées pour améliorer la recherche, afficher les sources, justifier la réponse et créer des liens vers la BDPM.

### Q3. Comment distinguer effets secondaires, posologie et interactions ?

Le projet utilise une fonction de détection d’intention.

Selon les mots présents dans la question, le système enrichit la recherche vectorielle avec des termes liés à la section médicale attendue.

Exemples :

- `effets indésirables` → section `4.8` ;
- `posologie` → section `4.2` ;
- `contre-indications` → section `4.3` ;
- `interactions` → section `4.5`.

### Q4. Comment gérer une question sur deux médicaments ?

Si l’utilisateur pose une question sur deux médicaments, FAISS recherche les passages proches de toute la question.

Le modèle Groq ne répond qu’à partir des passages retrouvés.

Si les sources ne contiennent pas d’information suffisante sur l’association des médicaments, le système doit répondre qu’il ne trouve pas l’information dans la base.

### Q5. Comment formuler un prompt prudent ?

Le prompt système impose plusieurs règles :

- ne pas inventer ;
- ne pas utiliser les connaissances générales ;
- ne pas donner de diagnostic ;
- citer les sources utilisées ;
- refuser si l’information est absente ;
- rappeler que l’assistant ne remplace pas un professionnel de santé.

---

## 28. Fichiers ignorés par Git

Le fichier `.gitignore` empêche de publier :

```text
venv/
.venv/
.env
data/*.zip
data/*.txt
data/*.csv
data/bdpm_raw/
data/medicaments_corpus_bdpm.json
__pycache__/
storage/
```

Pour le déploiement Streamlit, certains fichiers de `storage/` peuvent être ajoutés volontairement avec :

```powershell
git add -f storage/medicaments.index storage/chunks_medicaments.json storage/index_config.json
```

Cela est utile si l’on veut éviter de refaire l’indexation en ligne.

---

## 29. Problèmes rencontrés et solutions

### Problème 1 : `ModuleNotFoundError: No module named 'bs4'`

Solution :

```powershell
pip install beautifulsoup4
```

ou vérifier que `beautifulsoup4` est bien dans `requirements.txt`.

### Problème 2 : fichier `CIS_RCP.zip` introuvable

Solution : placer le fichier dans :

```text
data/CIS_RCP.zip
```

ou adapter le nom dans le script si le fichier téléchargé s’appelle autrement, par exemple :

```text
cis-rcp.zip
```

### Problème 3 : texte avec caractères mal encodés

Exemple :

```text
Ã©
```

Solution : tester plusieurs encodages et utiliser celui qui donne un texte lisible.

Le script teste notamment :

```text
utf-8-sig
utf-8
latin-1
```

### Problème 4 : erreur Groq 413 / Request too large

Cette erreur signifie que le contexte envoyé au modèle est trop grand.

Solutions :

- diminuer le nombre de sources FAISS ;
- diminuer la taille maximale par source ;
- envoyer uniquement les extraits pertinents ;
- réduire `max_tokens`.

### Problème 5 : secrets absents sur Streamlit Cloud

En ligne, le fichier `.env` n’est pas utilisé.

Il faut ajouter les variables dans les secrets Streamlit :

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

## 30. Limites du projet

Le projet présente certaines limites :

- la qualité des réponses dépend des données indexées ;
- si un médicament n’est pas dans le corpus, le chatbot ne peut pas répondre ;
- si une section est mal extraite, certaines informations peuvent manquer ;
- la recherche vectorielle peut récupérer des médicaments proches mais non exacts ;
- le modèle Groq peut parfois reformuler de manière imparfaite ;
- l’assistant ne remplace jamais un professionnel de santé.

---

## 31. Améliorations possibles

Plusieurs améliorations peuvent être ajoutées :

- ajouter une recherche hybride : FAISS + mots-clés ;
- ajouter un filtre strict par nom de médicament ;
- indexer toute la BDPM ;
- améliorer le découpage des sections ;
- ajouter une page d’accueil plus détaillée ;
- ajouter un historique de conversation ;
- ajouter un bouton d’export PDF ;
- ajouter un tableau comparatif entre deux médicaments ;
- ajouter une recherche par substance active ;
- ajouter un mode sans LLM affichant uniquement les extraits officiels ;
- améliorer encore le design Streamlit.

---

## 32. Commandes principales

Installation :

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Importer les données :

```powershell
python importer_bdpm.py --limit 1000
```

Créer l’index FAISS :

```powershell
python indexation.py --corpus data/medicaments_corpus_bdpm.json
```

Lancer le chatbot terminal :

```powershell
python rag.py
```

Lancer Streamlit :

```powershell
streamlit run app_streamlit.py
```

---

## 33. Auteur

Projet réalisé par :

```text
Billal Messaoui
assem el abrak 
```

Dans le cadre d’un TP sur la création d’un agent IA basé sur le RAG.
