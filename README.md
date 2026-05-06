# TP RAG — Assistant Médicaments avec BDPM, FAISS, Groq et Streamlit

Ce projet implémente un système **RAG complet** (*Retrieval-Augmented Generation*) pour répondre à des questions sur les médicaments à partir d’une base de connaissances construite avec des données issues de la **Base de Données Publique des Médicaments (BDPM)**.

Le projet correspond au **Sujet 2 : Assistant Médicaments** du TP. Il propose à la fois une version terminal et une interface web professionnelle avec **Streamlit**.

> **Important :** ce projet n’utilise ni LangChain ni LlamaIndex. Le pipeline RAG est implémenté manuellement en Python.

---

## 1. Objectif du projet

L’objectif est de construire un assistant capable de répondre à des questions comme :

```text
Quels sont les effets indésirables de l’amoxicilline ?
Quelle est la posologie de l’amoxicilline ?
Quelles sont les contre-indications de l’ibuprofène ?
Quels sont les effets secondaires du Doliprane ?
Puis-je prendre Doliprane et ibuprofène en même temps ?
```

L’assistant ne répond pas à partir de ses connaissances générales. Il répond uniquement à partir des sources récupérées dans la base vectorielle FAISS.

Le projet couvre les étapes suivantes :

- chargement d’une base documentaire réelle issue de la BDPM ;
- lecture de fichiers ZIP, CSV et TXT ;
- nettoyage du texte contenant du HTML ;
- extraction des sections médicales utiles ;
- construction d’un corpus JSON structuré ;
- découpage en chunks ;
- calcul des embeddings ;
- création d’un index FAISS persistant ;
- recherche sémantique et recherche hybride ;
- appel du modèle Groq ;
- génération de réponses sourcées ;
- affichage des sources officielles ;
- interface Streamlit déployée en ligne.

---

## 2. Application en ligne

L’application est déployée avec **Streamlit Community Cloud**.

Lien de l’application :

```text
https://cr-ationdagentiapourlapr-dictiondem-dicament-mqbcfubdiw83s7d2t.streamlit.app/
```

Cette interface permet d’utiliser le RAG comme un vrai chatbot, sans passer par le terminal.

---

## 3. Avertissement médical

Ce projet est un projet pédagogique. Les réponses générées ne remplacent jamais l’avis d’un médecin, d’un pharmacien ou d’un professionnel de santé.

L’assistant rappelle systématiquement :

```text
Ces informations ne remplacent pas l'avis d'un professionnel de santé.
En cas de doute, consultez votre médecin ou votre pharmacien.
```

Le chatbot ne donne pas de diagnostic médical et ne doit pas être utilisé pour prendre une décision de santé.

---

## 4. Source des données

Les données utilisées proviennent de la **Base de Données Publique des Médicaments (BDPM)**.

Les fichiers utilisés sont principalement :

```text
CIS_RCP.zip
CIS_bdpm_officielle.zip
CIS_COMPO_bdpm.txt
```

Ces fichiers permettent de récupérer :

- le code CIS des médicaments ;
- la dénomination des médicaments ;
- les substances actives ;
- la composition ;
- les résumés des caractéristiques du produit ;
- les indications ;
- la posologie ;
- les contre-indications ;
- les interactions ;
- les effets indésirables ;
- les informations de surdosage.

Contrairement à une première version basée sur un petit corpus local, cette version utilise une base réelle issue de la BDPM.

---

## 5. Problème rencontré avec les fichiers BDPM

Les fichiers BDPM ne sont pas directement simples à utiliser dans un système RAG.

Les principaux problèmes rencontrés sont :

- fichiers compressés au format `.zip` ;
- fichiers `.csv` ou `.txt` volumineux ;
- encodage parfois en `latin-1` ;
- encodage parfois en `utf-8-sig` ;
- séparateur pouvant être une tabulation, un point-virgule ou une virgule ;
- contenu médical stocké avec des balises HTML ;
- très grand nombre de lignes dans le fichier `CIS_RCP`.

Le fichier RCP peut être très lourd. Il ne faut donc pas le lire naïvement en mémoire sans précaution.

---

## 6. Solution utilisée pour lire et nettoyer les données

Le script `importer_bdpm.py` automatise la préparation des données.

### 6.1 Décompression automatique

Les fichiers `.zip` sont automatiquement décompressés dans :

```text
data/bdpm_raw/
```

Cela évite de décompresser manuellement les fichiers BDPM.

### 6.2 Détection de l’encodage

Le script teste plusieurs encodages :

```text
utf-8-sig
utf-8
latin-1
```

Cela permet d’éviter les problèmes de caractères mal lus, par exemple `Ã©`, `Ã¨` ou `Ãª`.

### 6.3 Détection du séparateur

Le script teste plusieurs séparateurs :

```text
tabulation
point-virgule
virgule
```

Le séparateur le plus cohérent est ensuite utilisé pour lire correctement le tableau.

### 6.4 Lecture par blocs

Les fichiers volumineux sont lus progressivement, notamment grâce à `chunksize` avec pandas.

Cette méthode permet de traiter de grands fichiers sans saturer la mémoire.

### 6.5 Nettoyage HTML

Les RCP contiennent souvent du texte avec des balises HTML. Le projet utilise **BeautifulSoup** pour transformer ce contenu en texte exploitable.

Le nettoyage consiste à :

- supprimer les balises HTML ;
- supprimer les balises inutiles comme `script` et `style` ;
- récupérer le texte lisible ;
- normaliser les espaces ;
- conserver les sections médicales importantes.

---

## 7. Sections médicales extraites

Le script extrait les sections importantes du RCP, par exemple :

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

Cette structuration améliore la précision du RAG. Par exemple, une question sur la posologie doit prioritairement récupérer la section `4.2`, tandis qu’une question sur les effets indésirables doit récupérer la section `4.8`.

---

## 8. Architecture du projet

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

## 9. Rôle des fichiers principaux

### `importer_bdpm.py`

Ce fichier prépare les données BDPM.

Il réalise notamment :

- la décompression des fichiers ZIP ;
- la lecture des fichiers BDPM ;
- la détection de l’encodage ;
- la détection du séparateur ;
- le nettoyage du HTML ;
- l’extraction des sections RCP ;
- le regroupement des informations par médicament ;
- la génération du corpus JSON.

Fichier généré :

```text
data/medicaments_corpus_bdpm.json
```

### `indexation.py`

Ce fichier crée la base vectorielle.

Il réalise :

- le chargement du corpus JSON ;
- la transformation des sections en documents ;
- le découpage en chunks ;
- le calcul des embeddings ;
- la normalisation des vecteurs ;
- la création d’un index FAISS ;
- la sauvegarde de l’index et des chunks.

Fichiers générés :

```text
storage/medicaments.index
storage/chunks_medicaments.json
storage/index_config.json
```

### `rag.py`

Ce fichier permet d’utiliser l’assistant dans le terminal.

Il réalise :

- le chargement de l’index FAISS ;
- le chargement des chunks ;
- le chargement du modèle d’embedding ;
- la vectorisation de la question ;
- la recherche des chunks pertinents ;
- la construction du contexte ;
- l’appel au modèle Groq ;
- l’affichage de la réponse ;
- l’affichage des sources récupérées.

### `app_streamlit.py`

Ce fichier contient l’interface web.

Il permet :

- de poser une question dans une interface graphique ;
- de filtrer par médicament ;
- de rechercher par substance active ;
- d’utiliser une recherche hybride FAISS + mots-clés ;
- de comparer deux médicaments ;
- de consulter l’historique de conversation ;
- d’afficher les sources officielles ;
- d’ouvrir les fiches BDPM ;
- d’exporter les réponses en TXT ou PDF.

### `requirements.txt`

Ce fichier contient les dépendances nécessaires.

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

Il ne faut pas écrire de commandes comme `pip install streamlit` dans `requirements.txt`. Il faut uniquement mettre le nom des bibliothèques.

---

## 10. Fonctionnalités implémentées

### 10.1 Recherche sémantique avec FAISS

La question est transformée en vecteur avec un modèle `sentence-transformers`, puis comparée aux vecteurs des chunks dans FAISS.

### 10.2 Recherche hybride FAISS + mots-clés

La version finale ajoute une recherche hybride. Elle combine :

- le score sémantique FAISS ;
- un score basé sur les mots-clés importants de la question ;
- un bonus selon l’intention détectée ;
- un bonus si la section médicale attendue correspond.

Cette amélioration aide à mieux récupérer les sections importantes comme `4.2 Posologie`, `4.3 Contre-indications`, `4.5 Interactions` ou `4.8 Effets indésirables`.

### 10.3 Filtre strict par médicament

L’utilisateur peut sélectionner un médicament précis dans l’interface.

Le filtre strict permet d’éviter que le système récupère un médicament proche mais différent.

### 10.4 Recherche par substance active

L’interface permet aussi de rechercher par substance active, par exemple :

```text
amoxicilline
ibuprofène
paracétamol
```

### 10.5 Détection d’intention

Le système détecte l’intention de la question :

- effets indésirables ;
- posologie ;
- contre-indications ;
- interactions ;
- information générale.

Cette intention est utilisée pour enrichir la recherche et améliorer la pertinence des sources.

### 10.6 Historique de conversation

L’interface Streamlit conserve un historique des questions posées et des réponses générées pendant la session.

### 10.7 Export TXT et PDF

L’utilisateur peut télécharger :

- la réponse en `.txt` ;
- la réponse en `.pdf` ;
- une comparaison entre deux médicaments en PDF.

### 10.8 Tableau comparatif entre deux médicaments

L’application permet de comparer deux médicaments sur un critère précis :

- effets indésirables ;
- posologie ;
- contre-indications ;
- interactions ;
- information générale.

La comparaison est générée uniquement à partir des sources récupérées.

### 10.9 Liens vers les sources officielles BDPM

Chaque source récupérée affiche un lien vers la fiche officielle BDPM lorsque le code CIS est disponible.

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

Importer un nombre limité de médicaments :

```powershell
python importer_bdpm.py --limit 1000
```

Importer davantage de médicaments :

```powershell
python importer_bdpm.py --limit 10000
```

Importer tous les médicaments :

```powershell
python importer_bdpm.py --limit 0
```

Filtrer sur un médicament précis :

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
Quelle est la posologie de l’amoxicilline ?
Quelles sont les contre-indications de l’ibuprofène ?
C’est quoi le Doliprane ?
```

Pour quitter :

```text
quit
exit
q
```

---

## 16. Lancer l’interface Streamlit en local

```powershell
streamlit run app_streamlit.py
```

L’application s’ouvre ensuite dans le navigateur :

```text
http://localhost:8501
```

---

## 17. Déploiement avec Streamlit Cloud

Le projet peut être déployé avec **Streamlit Community Cloud**.

Étapes :

1. publier le projet sur GitHub ;
2. aller sur Streamlit Community Cloud ;
3. créer une nouvelle application ;
4. choisir le dépôt GitHub ;
5. choisir la branche `main` ;
6. choisir le fichier principal `app_streamlit.py` ;
7. ajouter les variables secrètes dans les secrets Streamlit ;
8. lancer le déploiement.

---

## 18. Gestion de la clé API

Le fichier `.env` n’est pas envoyé sur GitHub pour des raisons de sécurité.

En local, la clé est lue depuis `.env`.

En ligne, la clé est lue depuis les **Secrets** de Streamlit Cloud.

Exemple dans Streamlit Secrets :

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

La question de l’utilisateur est aussi transformée en vecteur. FAISS compare ensuite le vecteur de la question avec les vecteurs des chunks. Les chunks les plus proches sont récupérés et envoyés au modèle Groq.

Cela évite d’envoyer toute la base de données au LLM.

---

## 20. Base vectorielle FAISS

Le projet utilise FAISS pour stocker et rechercher les embeddings.

Les embeddings sont normalisés. L’index FAISS permet donc de comparer les textes avec une similarité proche de la similarité cosinus.

L’index est sauvegardé dans :

```text
storage/medicaments.index
```

Cela évite de recalculer les embeddings à chaque lancement du chatbot.

---

## 21. Modèle Groq utilisé

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

## 22. Mécanismes anti-hallucination

Le projet contient plusieurs mécanismes pour limiter les hallucinations.

### 22.1 Réponse basée uniquement sur le contexte

Le prompt système interdit au modèle d’utiliser ses connaissances générales.

### 22.2 Score de confiance

Chaque source récupérée possède un score. Si le score est trop faible, l’assistant refuse de répondre au lieu d’inventer.

### 22.3 Vérification des termes importants

Le système extrait les mots importants de la question. Si un médicament ou un terme important n’apparaît pas dans les sources récupérées, le système peut refuser de répondre.

### 22.4 Réduction du contexte envoyé au LLM

Pour éviter les erreurs de limite de tokens, l’application n’envoie pas tout le contenu des sources au modèle. Elle extrait seulement les passages les plus pertinents autour des mots-clés de la question.

### 22.5 Citation des sources

Les réponses doivent citer les sources utilisées avec :

- le nom du médicament ;
- la section ;
- le code CIS ;
- le lien officiel BDPM quand il est disponible.

---

## 23. Interface Streamlit finale

L’interface Streamlit contient :

- une page d’accueil claire ;
- un chatbot RAG ;
- une recherche hybride ;
- un filtre strict par médicament ;
- une recherche par substance active ;
- une page de comparaison entre deux médicaments ;
- une page de recherche dans les sources ;
- un historique de conversation ;
- un export TXT et PDF ;
- des liens vers les fiches officielles BDPM ;
- l’affichage des scores et des extraits utilisés.

L’objectif est de rendre le projet facile à tester et plus professionnel pour la démonstration.

---

## 24. Exemple de réponse

Question :

```text
Quels sont les effets indésirables de l’amoxicilline ?
```

Exemple de réponse attendue :

```text
Les effets indésirables mentionnés dans les sources récupérées peuvent inclure :

- troubles digestifs ;
- nausées ;
- diarrhée ;
- réactions cutanées ;
- manifestations allergiques.

Ces informations ne remplacent pas l'avis d'un professionnel de santé.
En cas de doute, consultez votre médecin ou votre pharmacien.
```

Les sources affichées permettent ensuite de vérifier les informations :

```text
Médicament : AMOXICILLINE ...
Section : 4.8 Effets indésirables
Code CIS : ...
Score : ...
Lien BDPM : ...
```

---

## 25. Réponses aux questions de réflexion du sujet

### Q1. Quelle stratégie de chunking pour les notices longues ?

Les notices sont longues et contiennent plusieurs sections médicales. La stratégie utilisée consiste à extraire les sections importantes du RCP, puis à découper ces sections en chunks.

Cela permet de ne pas envoyer toute une notice au LLM.

### Q2. Comment exploiter la structure des notices ?

Les notices/RCP contiennent des sections officielles. Le projet conserve ces sections dans les métadonnées :

```json
{
  "medicament": "AMOXICILLINE ...",
  "cis": "66082724",
  "section": "4.8 Effets indésirables",
  "source": "BDPM - CIS_RCP.zip"
}
```

Ces métadonnées sont utilisées pour améliorer la recherche, afficher les sources et justifier la réponse.

### Q3. Comment distinguer effets secondaires, posologie et interactions ?

Le projet utilise une fonction de détection d’intention. Selon les mots présents dans la question, le système enrichit la recherche vectorielle avec des termes liés à la section attendue.

Exemples :

- `effets indésirables` → section `4.8` ;
- `posologie` → section `4.2` ;
- `contre-indications` → section `4.3` ;
- `interactions` → section `4.5`.

### Q4. Comment gérer une question sur deux médicaments ?

L’application propose un onglet de comparaison entre deux médicaments. Le système récupère les sources pertinentes pour chaque médicament, puis demande au modèle Groq de produire une réponse comparative uniquement à partir des sources récupérées.

Si les sources ne contiennent pas d’information suffisante, le système doit le signaler.

### Q5. Comment formuler un prompt prudent ?

Le prompt système impose au modèle de :

- ne pas inventer ;
- ne pas utiliser les connaissances générales ;
- ne pas donner de diagnostic ;
- citer les sources utilisées ;
- refuser si l’information est absente ;
- rappeler que l’assistant ne remplace pas un professionnel de santé.

---

## 26. Fichiers ignorés par Git

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
```

Pour le déploiement Streamlit, certains fichiers générés dans `storage/` peuvent être ajoutés volontairement avec :

```powershell
git add -f storage/medicaments.index storage/chunks_medicaments.json storage/index_config.json
```

Cela permet à l’application en ligne de charger directement l’index sans refaire l’indexation.

---

## 27. Problèmes rencontrés et solutions

### Problème 1 : `ModuleNotFoundError: No module named 'bs4'`

Solution :

```powershell
pip install beautifulsoup4
```

ou vérifier que `beautifulsoup4` est bien dans `requirements.txt`.

### Problème 2 : fichier BDPM introuvable

Solution : placer les fichiers BDPM dans :

```text
data/
```

Exemples :

```text
data/CIS_RCP.zip
data/CIS_bdpm_officielle.zip
data/CIS_COMPO_bdpm.txt
```

### Problème 3 : texte avec caractères mal encodés

Exemple :

```text
Ã©
```

Solution : tester plusieurs encodages et utiliser celui qui donne un texte lisible.

### Problème 4 : erreur Groq 413 / Request too large

Cette erreur signifie que le contexte envoyé au modèle est trop grand.

Solutions :

- diminuer le nombre de sources ;
- diminuer la taille maximale par source ;
- envoyer uniquement les extraits pertinents ;
- réduire `max_tokens`.

### Problème 5 : secrets absents sur Streamlit Cloud

En ligne, le fichier `.env` n’est pas utilisé. Il faut ajouter les variables dans les secrets Streamlit :

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_MODEL = "llama-3.1-8b-instant"
```

### Problème 6 : erreur dans `requirements.txt`

Il ne faut pas écrire :

```text
pip install streamlit
```

dans `requirements.txt`.

Il faut écrire seulement :

```text
streamlit
```

---

## 28. Limites du projet

Le projet présente certaines limites :

- la qualité des réponses dépend des données indexées ;
- si un médicament n’est pas dans le corpus, le chatbot ne peut pas répondre ;
- si une section est mal extraite, certaines informations peuvent manquer ;
- la recherche vectorielle peut récupérer des médicaments proches ;
- le modèle Groq peut reformuler de manière imparfaite ;
- l’assistant ne remplace jamais un professionnel de santé.

---

## 29. Améliorations futures possibles

Plusieurs améliorations peuvent encore être ajoutées :

- indexer une version encore plus complète de la BDPM ;
- améliorer le découpage automatique des sections RCP ;
- ajouter un mode de recherche avancée multicritère ;
- ajouter une évaluation automatique de la qualité des réponses ;
- ajouter un système de feedback utilisateur ;
- ajouter une authentification pour un usage privé ;
- ajouter une base vectorielle distante pour éviter de stocker l’index dans GitHub.

---

## 30. Commandes principales

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

## 31. Auteur

Projet réalisé par :

```text
Billal Messaoui
Assem El Abrak
```

Dans le cadre d’un TP sur la création d’un agent IA basé sur le RAG.
