from pathlib import Path
from datetime import datetime
import argparse
import csv
import subprocess

import imageio_ffmpeg
import numpy as np
from scipy.signal import stft, correlate, find_peaks


DOSSIER = Path(__file__).resolve().parent

VIDEO_PAR_DEFAUT = Path(
    r"C:\Users\ASUS\Downloads"
    r"\a96a1c94-7277-4024-8a00-fa0133934157.mp4"
)

FS = 16000
PAS = 160  # Comparaison toutes les 0,01 seconde.

REFERENCES = [
    DOSSIER / f"{prefixe}{numero}.wav"
    for prefixe in (
        "reference_lame_",
        "video2_reference_lame_",
    )
    for numero in (1, 2, 3)
]


def lire_audio(chemin):
    """Extraire le son en mono, à 16 kHz."""
    chemin = Path(chemin)

    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier absent : {chemin}")

    commande = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v", "error",
        "-i", str(chemin),
        "-vn",
        "-ac", "1",
        "-ar", str(FS),
        "-f", "f32le",
        "pipe:1",
    ]

    resultat = subprocess.run(
        commande,
        capture_output=True,
        check=True,
    )

    audio = np.frombuffer(
        resultat.stdout, dtype="<f4"
    ).copy()

    if len(audio) < 400:
        raise ValueError(f"Audio absent ou trop court : {chemin.name}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Audio invalide : {chemin.name}")

    if np.max(np.abs(audio)) < 1e-7:
        raise ValueError(f"Audio silencieux : {chemin.name}")

    return audio


def signature(audio):
    """Décrire les variations d'énergie par fréquence."""
    frequences, _, spectre = stft(
        audio,
        fs=FS,
        nperseg=400,
        noverlap=400 - PAS,
        nfft=512,
        boundary=None,
        padded=False,
    )

    puissance = np.abs(spectre) ** 2
    limites = np.geomspace(150, 7500, 25)
    bandes = []

    for bas, haut in zip(limites[:-1], limites[1:]):
        selection = (
            (frequences >= bas)
            & (frequences < haut)
        )
        bandes.append(
            puissance[selection].mean(axis=0)
        )

    return np.log10(
        np.maximum(np.array(bandes), 1e-12)
    )


def comparer(signal, modele):
    """Calculer un score de ressemblance à chaque position."""
    largeur = modele.shape[1]

    if largeur > signal.shape[1]:
        raise ValueError(
            "Une reference est plus longue que la video."
        )

    modele = modele - modele.mean(
        axis=1, keepdims=True
    )

    energie_modele = np.sum(modele ** 2)

    if energie_modele < 1e-10:
        raise ValueError(
            "Reference sans variations sonores suffisantes."
        )

    taille = signal.shape[1] - largeur + 1
    numerateur = np.zeros(taille)
    energie = np.zeros(taille)
    fenetre = np.ones(largeur)

    for bande, motif in zip(signal, modele):
        numerateur += correlate(
            bande, motif, mode="valid"
        )

        somme = np.convolve(
            bande, fenetre, mode="valid"
        )

        somme_carres = np.convolve(
            bande ** 2, fenetre, mode="valid"
        )

        energie += np.maximum(
            somme_carres - somme ** 2 / largeur,
            0,
        )

    denominateur = np.sqrt(
        np.maximum(energie * energie_modele, 1e-20)
    )

    return np.clip(
        numerateur / denominateur, -1, 1
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test de comptage audio avec six references."
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PAR_DEFAUT,
    )
    parser.add_argument(
        "--seuil",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--ecart",
        type=float,
        default=0.22,
        help="Ecart minimal entre deux detections, en secondes.",
    )
    parser.add_argument(
        "--attendu",
        type=int,
        default=None,
        help="Nombre reel de cables, uniquement pour comparaison.",
    )

    args = parser.parse_args()

    if not 0 < args.seuil <= 1:
        raise ValueError("Le seuil doit etre entre 0 et 1.")

    if args.ecart <= 0:
        raise ValueError("L'ecart doit etre positif.")

    if args.attendu is not None and args.attendu < 0:
        raise ValueError("Le nombre attendu doit etre positif ou nul.")

    # Vérifier les six fichiers avant l'analyse.
    manquantes = [
        str(fichier)
        for fichier in REFERENCES
        if not fichier.is_file()
    ]

    if manquantes:
        raise FileNotFoundError(
            "References absentes :\n" + "\n".join(manquantes)
        )

    print(f"\nVideo : {args.video}")
    print(f"Seuil : {args.seuil}")
    print(f"Ecart anti-doublon : {args.ecart} s")
    print("\nLecture du son...")

    audio_video = lire_audio(args.video)
    signal = signature(audio_video)

    resultats = []
    durees = []

    for numero, fichier in enumerate(REFERENCES, 1):
        audio_reference = lire_audio(fichier)
        modele = signature(audio_reference)
        scores = comparer(signal, modele)

        resultats.append(scores)
        durees.append(len(audio_reference) / FS)

        print(
            f"Reference {numero} : {fichier.name}"
            f" | score maximal {scores.max():.3f}"
        )

    # Aligner les scores des six références sur le temps.
    taille = max(len(scores) for scores in resultats)

    matrice = np.full(
        (len(resultats), taille),
        -np.inf,
    )

    for index, scores in enumerate(resultats):
        matrice[index, :len(scores)] = scores

    # Un seul score par instant :
    # on garde la meilleure correspondance, sans additionner.
    meilleurs_scores = matrice.max(axis=0)
    meilleures_refs = matrice.argmax(axis=0)

    # Supprimer les pics trop proches.
    # Ce réglage limite les doublons mais reste à valider.
    distance = max(
        1,
        int(np.ceil(args.ecart * FS / PAS)),
    )

    pics, _ = find_peaks(
        np.r_[-np.inf, meilleurs_scores, -np.inf],
        height=args.seuil,
        distance=distance,
    )
    pics = pics - 1

    # Un fichier distinct par exécution conserve les anciens tests.
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sortie = DOSSIER / f"detections_6refs_{horodatage}.csv"

    print("\n--- EVENEMENTS CANDIDATS ---")

    with sortie.open(
        "w", newline="", encoding="utf-8-sig"
    ) as fichier:
        writer = csv.writer(fichier, delimiter=";")

        writer.writerow([
            "evenement",
            "debut_secondes",
            "fin_secondes",
            "reference",
            "similarite",
            "cables_estimes_cumules",
            "evenements_restants",
        ])

        for numero, pic in enumerate(pics, 1):
            ref = int(meilleures_refs[pic])
            debut = pic * PAS / FS
            fin = debut + durees[ref]
            score = float(meilleurs_scores[pic])

            cables = numero // 3
            reste = numero % 3

            writer.writerow([
                numero,
                f"{debut:.3f}",
                f"{fin:.3f}",
                REFERENCES[ref].name,
                f"{score:.3f}",
                cables,
                reste,
            ])

            print(
                f"Evenement {numero:3d}"
                f" | {debut:6.2f} s"
                f" | reference {ref + 1}"
                f" | similarite {score:.3f}"
                f" | cables estimes {cables}"
            )

    total = len(pics)

    print("\n--- RESULTAT PROVISOIRE ---")
    print(f"Duree audio : {len(audio_video) / FS:.2f} s")
    print(f"References utilisees : {len(REFERENCES)}")
    print(f"Evenements candidats : {total}")
    print(f"Cables estimes (division par 3) : {total // 3}")
    print(f"Evenements restants : {total % 3}")

    if args.attendu is not None:
        print(f"Reference manuelle : {args.attendu} cables")
        print(
            "Evenements attendus pour des cycles complets : "
            f"{args.attendu * 3}"
        )

    print("\nLe nombre attendu ne modifie pas la detection.")
    print("La similarite n'est pas une probabilite.")
    print("Verifier les instants detectes, pas seulement le total.")
    print(f"\nCSV : {sortie}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as erreur:
        message = erreur.stderr.decode(
            "utf-8", errors="replace"
        )
        print("Erreur FFmpeg :", message)
        raise SystemExit(1)
    except Exception as erreur:
        print("Erreur :", erreur)
        raise SystemExit(1)