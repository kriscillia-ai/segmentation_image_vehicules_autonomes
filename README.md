# Segmentation d'Images pour Véhicules Autonomes : Du Modèle Deep Learning au Déploiement Cloud

Ce projet présente le développement d'une application de segmentation d'images de bout en bout, conçue pour améliorer la perception des **véhicules autonomes**. L'objectif est de segmenter avec précision des scènes urbaines en identifiant des objets clés tels que les routes, les voitures, les piétons et les bâtiments, en utilisant le **Deep Learning** et un déploiement sur le **cloud**.

L'application combine un modèle de segmentation sémantique basé sur une architecture **VGG16 U-Net**, une **API Flask** pour les prédictions en temps réel et une **interface utilisateur Streamlit** pour la visualisation des résultats. L'ensemble est déployé sur **Google Cloud Platform (GCP)**.

## Algorithme et Modélisation

Le cœur de ce projet repose sur une architecture **VGG16 U-Net**.

  - **VGG16 U-Net** : Une combinaison de l'encodeur **VGG16 pré-entraîné** pour l'extraction de caractéristiques et du décodeur **U-Net** pour une segmentation précise des pixels. Le modèle a été entraîné sur un ensemble de données de scènes urbaines avec des **augmentations de données** pour améliorer sa robustesse.
  - **Générateurs de Données Personnalisés (Sequence)** : Optimisation de l'entraînement du modèle par la mise en œuvre de générateurs de données permettant un chargement efficace et une augmentation en temps réel.
  - **Albumentations** : Utilisation de cette bibliothèque pour des augmentations de données avancées (retournements, rotations, ajustements de luminosité/contraste) afin de renforcer la généralisation du modèle.

## Fonctionnalités Principales

  - **Segmentation Sémantique** : Le modèle segmente les scènes urbaines en 8 catégories clés : routes, trottoirs, bâtiments, murs, clôtures, poteaux/feux de circulation, végétation et terrain.
  - **API RESTful (Flask)** : Une API a été développée pour exposer le modèle de segmentation. Elle gère les requêtes de prédiction en temps réel et s'intègre avec **Google Cloud Storage** pour le stockage des images.
  - **Interface Utilisateur (Streamlit)** : Une application interactive permet aux utilisateurs de soumettre une image et de visualiser côte à côte l'image originale, le masque réel et le masque prédit par le modèle.
  - **Déploiement Cloud (Google Cloud)** :
      - L'API Flask et l'application Streamlit sont conteneurisées avec **Docker** et déployées sur **Google Cloud Run** pour une scalabilité et une gestion simplifiée.
      - **Google Cloud Storage** est utilisé pour un stockage fiable et évolutif des données, des modèles et des résultats de segmentation.

## Technologies Utilisées

| Catégorie | Outils & Bibliothèques |
| :--- | :--- |
| **Deep Learning** | `TensorFlow`, `Keras` |
| **Cloud Computing** | `Google Cloud Run`, `Google Cloud Storage` |
| **Développement Web** | `Flask`, `Streamlit` |
| **Traitement d'Images** | `OpenCV`, `PIL`, `Albumentations` |
| **Langage** | `Python` |
| **Outils** | `Docker`, `Git` |

## Démonstration

<img width="651" height="212" alt="image" src="https://github.com/user-attachments/assets/88d62ab2-ffd5-4a54-84f7-56c294cc4ccd" />
<img width="671" height="218" alt="image" src="https://github.com/user-attachments/assets/4ce29a25-01cf-40f6-a3da-25b45be526fe" />
<img width="523" height="240" alt="image" src="https://github.com/user-attachments/assets/4b855254-6034-4d8c-bf83-63d012143849" />


## Défis Relevés

  - **Gestion de grands ensembles de données** d'images et de masques.
  - **Optimisation du modèle** pour la performance en temps réel, cruciale pour les applications embarquées.
  - **Conteneurisation et déploiement** d'applications sur une plateforme serverless comme Google Cloud Run.
