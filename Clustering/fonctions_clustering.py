# Ici sera le code avec les fonctions utiles de l'algo de clustering
# Nous appelons ce fichier dans le notebook final afin de ne pas surcharger le notebook

# Importer les bibliothèques nécessaires
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from glob import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Charger ResNet50 sans la couche de classification
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Chemin vers le dossier d'images
image_paths = glob("../DATASET_SENSEA/images/*.jpg")    # Le .jpg à la fin permet de prendre uniquement les photos et pas les autres fichiers
    


def extract_features(img_path, model):
    # Charger l'image et la redimensionner à 224x224 pixels
    img = image.load_img(img_path, target_size=(224, 224))
    # Convertir l'image en un tableau numpy
    x = image.img_to_array(img)
    # Ajouter une dimension (pour représenter le batch de 1 image)
    x = np.expand_dims(x, axis=0)
    # Prétraiter l'image pour ResNet
    x = preprocess_input(x)
    # Passer l'image dans le modèle pour extraire les features
    features = model.predict(x)
    # Aplatir le vecteur de sortie (c'est déjà un vecteur de 2048, donc facultatif ici)
    return features.flatten()



# Cette fonction nous renvoie un tableau de features pour toutes les images, exploitable pour le clustering (avec KMeans)
def array_features(img_paths):
    # Initialiser une liste pour stocker tous les vecteurs de features
    features_list = []
    # Boucler sur chaque image et extraire les features
    for img_path in img_paths:
            features = extract_features(img_path, model)
            features_list.append(features)
    # Convertir la liste en un tableau numpy pour faciliter l'analyse
    features_array = np.array(features_list)
    return features_array




# L'array donnée par la fonction array_features() comporte 2048 features pour chaque image, on cherche à réduire ce nombre (à 50)
def pca_reduction(features_array):
    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(features_array)
    return reduced_features




def kmeans_clustering(reduced_features, img_paths):
    k_optimal = 5
    # Appliquer K-means
    kmeans = KMeans(n_clusters=k_optimal, random_state=0)
    labels_kmeans = kmeans.fit_predict(reduced_features)
    images = [image.load_img(img_path) for img_path in img_paths]
    return labels_kmeans, images




def afficher_photos_aleatoires(images, labels_kmeans, num_photos=50):
    """
    Affiche un ensemble de photos choisies aléatoirement avec leurs étiquettes de cluster.

    Paramètres :
    - images : liste ou tableau contenant les images (chaque image est un tableau).
    - labels : tableau des étiquettes de cluster correspondant à chaque image.
    - num_photos : nombre de photos à afficher (défaut 50).
    """
    # Sélectionne des indices aléatoires
    indices = np.random.choice(len(images), num_photos, replace=False)

    # Paramètres pour l'affichage
    nb_colonnes = 10  # Nombre de colonnes de l'affichage
    nb_lignes = num_photos // nb_colonnes + (num_photos % nb_colonnes > 0)

    # Crée une figure pour afficher les images
    plt.figure(figsize=(15, nb_lignes * 1.5))

    for i, idx in enumerate(indices):
        plt.subplot(nb_lignes, nb_colonnes, i + 1)
        plt.imshow(images[idx])  # Affiche l'image
        plt.title(f'Cluster {labels_kmeans[idx]}')  # Affiche l'étiquette
        plt.axis('off')  # Cache les axes pour une meilleure visibilité

    plt.tight_layout()
    plt.show()