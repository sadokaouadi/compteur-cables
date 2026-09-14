import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import cv2
import numpy as np


fenetre = tk.Tk()
fenetre.withdraw()

chemin = filedialog.askopenfilename(
    title="Choisir la vidéo de 7 câbles",
    filetypes=[("Vidéos", "*.mp4 *.mov *.avi")]
)

fenetre.destroy()

if not chemin:
    raise SystemExit("Aucune vidéo sélectionnée.")

capture = cv2.VideoCapture(chemin)

if not capture.isOpened():
    raise SystemExit("Impossible d'ouvrir la vidéo.")

# Ouvertures et fermetures au début, au milieu et à la fin.
numeros = [
    40, 43, 44, 45, 50,
    416, 418, 420, 437, 441,
    602, 604, 624, 625, 627,
]

planche = np.full((3 * 280, 5 * 200, 3), 245, dtype=np.uint8)

try:
    for index, numero in enumerate(numeros):
        capture.set(cv2.CAP_PROP_POS_FRAMES, numero)
        succes, image = capture.read()

        if not succes:
            raise RuntimeError(f"Image {numero} illisible.")

        hauteur, largeur = image.shape[:2]

        # Zone actuelle : horizontal 29–38 %, vertical 48–55 %.
        zone = image[
            int(hauteur * 0.48):int(hauteur * 0.55),
            int(largeur * 0.29):int(largeur * 0.38)
        ]

        facteur = min(180 / zone.shape[1], 240 / zone.shape[0])
        zoom = cv2.resize(
            zone,
            None,
            fx=facteur,
            fy=facteur,
            interpolation=cv2.INTER_NEAREST
        )

        x = (index % 5) * 200
        y = (index // 5) * 280

        cv2.putText(
            planche, f"Image {numero}", (x + 10, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        h, w = zoom.shape[:2]
        planche[y + 35:y + 35 + h, x + 10:x + 10 + w] = zoom

finally:
    capture.release()

sortie = Path(__file__).resolve().parent / "diagnostic_lame.png"

if not cv2.imwrite(str(sortie), planche):
    raise RuntimeError("Impossible d'enregistrer l'image.")

print(f"Image créée : {sortie}")