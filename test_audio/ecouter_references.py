import winsound
import time

for numero in (1, 2, 3):
    fichier = f"reference_lame_{numero}.wav"
    input(f"Reference {numero} : appuyez sur Entree pour ecouter.")
    for _ in range(3):
        winsound.PlaySound(
            fichier,
            winsound.SND_FILENAME | winsound.SND_NODEFAULT
        )
        time.sleep(1)
