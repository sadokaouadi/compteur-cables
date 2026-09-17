import sys
from pathlib import Path

import cv2
import zxingcpp


def lire_code_barres(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"ERREUR : image introuvable : {image_path}")
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERREUR : impossible d'ouvrir l'image.")
        return

    print("=" * 50)
    print("TEST CODE-BARRES")
    print("=" * 50)

    print(f"Image : {image_path}")
    print(f"Resolution : {image.shape[1]} x {image.shape[0]}")
    print()

    resultats = zxingcpp.read_barcodes(image)

    if not resultats:
        print("Aucun code-barres detecte.")
        return

    print(f"Nombre de codes trouves : {len(resultats)}")
    print()

    for i, resultat in enumerate(resultats, start=1):

        texte = resultat.text

        print(f"--- Code {i} ---")
        print(f"Valeur : {texte}")
        print(f"Format : {resultat.format}")

        # Validation STARZ : exemple 48247-1
        import re

        if re.fullmatch(r"\d{5}-\d", texte):
            print("Validation STARZ : OK")
        else:
            print("Validation STARZ : REJETE")

        print()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Utilisation :")
        print(
            r'.\venv\Scripts\python.exe test_barcode.py "image.png"'
        )
        sys.exit(1)

    lire_code_barres(sys.argv[1])