# Compte-rendu court — TP RAG Médicaments

## Sujet choisi

J'ai choisi le sujet B : **Assistant Médicaments**. L'objectif était de construire un système RAG capable de répondre à des questions en langage naturel à partir d'une base de connaissances locale sur des médicaments.

## Choix de conception

J'ai représenté chaque médicament sous forme structurée avec plusieurs sections : indications, posologie, contre-indications, effets indésirables et interactions. Ce choix permet de mieux contrôler la recherche vectorielle et d'aider le LLM à citer précisément ses sources.

Le chunking est réalisé avec une taille maximale de 900 caractères et un overlap de 120 caractères. Ce compromis permet de conserver assez de contexte médical sans envoyer des passages trop longs au modèle Groq.

Pour les embeddings, j'ai utilisé `paraphrase-multilingual-mpnet-base-v2`, car le corpus et les questions sont en français. Les vecteurs sont normalisés et indexés avec `faiss.IndexFlatIP`, ce qui revient à utiliser une similarité cosinus.

## Persistance

L'index FAISS est sauvegardé dans `storage/medicaments.index` et les chunks avec métadonnées dans `storage/chunks_medicaments.json`. Ainsi, l'indexation n'est exécutée qu'une seule fois avec `python indexation.py`. Le script `rag.py` recharge ensuite directement la base vectorielle.

## Sécurité et prudence médicale

Le prompt système impose au LLM de ne répondre qu'à partir du contexte fourni, de citer les médicaments et sections utilisés, et de refuser si l'information n'est pas présente dans la base. Chaque réponse doit inclure la phrase obligatoire :

> Ces informations ne remplacent pas l'avis d'un professionnel de santé.

## Difficultés rencontrées

La principale difficulté est que les notices de médicaments sont longues et très denses. Il faut éviter de mélanger plusieurs informations médicales dans un même chunk. Pour cette raison, le corpus est d'abord séparé par sections avant le chunking.

Une autre difficulté est la limitation du corpus : avec seulement 15 à 20 médicaments, certaines questions ne peuvent pas recevoir de réponse. Le système doit donc refuser d'inventer. J'ai ajouté un score de confiance pour signaler les recherches peu pertinentes.

## Amélioration possible

Pour améliorer le projet, il faudrait remplacer le corpus local par des extraits officiels directement issus du fichier `CIS_RCP.zip` de la Base de Données Publique des Médicaments. On pourrait aussi ajouter un mode de comparaison entre deux médicaments et un historique de conversation.
