import os
import nbformat
from nbconvert import PythonExporter

def convert_notebook_to_python(notebook_path, output_path=None):
    # Charger le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Convertir en script Python
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(notebook)

    # Définir le chemin de sortie
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(notebook_path))[0]
        output_path = f"{base_name}_converted.py"
    
    # Vérifier si le fichier existe déjà
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base_name}_converted_{counter}.py"
        counter += 1

    # Écrire le fichier Python
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(python_code)

    print(f"Notebook converti et enregistré sous : {output_path}")

# Exemple d'utilisation
convert_notebook_to_python("ClusteringComplet.ipynb")
