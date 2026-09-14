import tempfile
from pathlib import Path

import cv2
import streamlit as st


st.set_page_config(
    page_title="Diagnostic du suivi",
    layout="wide"
)

st.title("Diagnostic : câble et zone au-dessus")

st.info(
    "Test prévu pour la vidéo du câble jaune. "
    "Vert = câble repéré. Bleu = zone à observer au-dessus. "
    "Aucun comptage n'est effectué."
)

video = st.file_uploader(
    "Choisir la vidéo du câble jaune",
    type=["mp4", "mov", "avi"]
)

if video is None:
    st.stop()

capture = None
chemin_temp = None

try:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(video.name).suffix
    ) as fichier_temp:
        fichier_temp.write(video.getvalue())
        chemin_temp = fichier_temp.name

    capture = cv2.VideoCapture(chemin_temp)

    if not capture.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if total <= 0:
        raise ValueError("Impossible de déterminer le nombre d'images.")

    numero = int(st.number_input(
        "Numéro de l'image à examiner",
        min_value=0,
        max_value=total - 1,
        value=min(40, total - 1),
        step=1,
        key=f"image_suivi_{video.file_id}"
    ))

    capture.set(cv2.CAP_PROP_POS_FRAMES, numero)
    succes, image = capture.read()

    if not succes:
        raise ValueError(f"Impossible de lire l'image {numero}.")

    if fps > 0:
        st.caption(f"Position : {numero / fps:.2f} secondes")

    hauteur, largeur = image.shape[:2]

    st.subheader("Zone du mécanisme")

    gauche, droite = st.slider(
        "Limites horizontales (%)",
        0, 100, (29, 38),
        key=f"suivi_x_{video.file_id}"
    )

    haut, bas = st.slider(
        "Limites verticales (%)",
        0, 100, (48, 55),
        key=f"suivi_y_{video.file_id}"
    )

    x1 = int(largeur * gauche / 100)
    x2 = int(largeur * droite / 100)
    y1 = int(hauteur * haut / 100)
    y2 = int(hauteur * bas / 100)

    if x2 <= x1 or y2 <= y1:
        st.warning("Choisis une zone non vide.")
        st.stop()

    zone = image[y1:y2, x1:x2].copy()

    # Réglages de départ pour isoler la couleur jaune.
    # Ils restent à vérifier sur la vidéo.
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

    masque = cv2.inRange(
        hsv,
        (15, 70, 60),
        (40, 255, 255)
    )

    # Chercher les groupes de pixels jaunes.
    nombre, etiquettes, statistiques, centres = (
        cv2.connectedComponentsWithStats(
            masque,
            connectivity=8
        )
    )

    candidats = []

    for index in range(1, nombre):
        x, y, w, h, aire = statistiques[index]

        # Retirer les très petites taches.
        # Le câble attendu est plutôt horizontal.
        if aire >= 10 and w >= 5 and w >= h:
            candidats.append((
                int(aire),
                int(x),
                int(y),
                int(w),
                int(h)
            ))

    pleine_image = image.copy()

    cv2.rectangle(
        pleine_image,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 0),
        2
    )

    colonne1, colonne2 = st.columns(2)

    with colonne1:
        st.image(
            cv2.cvtColor(pleine_image, cv2.COLOR_BGR2RGB),
            caption=f"Image {numero} — zone du mécanisme",
            width=360
        )

    if not candidats:
        with colonne2:
            st.image(
                cv2.cvtColor(zone, cv2.COLOR_BGR2RGB),
                caption="Zone examinée",
                width=360
            )

        st.warning(
            "Aucune forme jaune correspondant aux critères "
            "n'a été trouvée dans cette zone."
        )

    else:
        # Hypothèse à vérifier visuellement :
        # la plus grande forme jaune horizontale est le câble.
        aire, x, y, w, h = max(candidats)

        st.subheader("Placement de la zone bleue")

        hauteur_bande = st.slider(
            "Hauteur de la zone bleue (pixels)",
            min_value=3,
            max_value=25,
            value=10
        )

        decalage = st.slider(
            "Décalage horizontal depuis le bord gauche du câble (pixels)",
            min_value=0,
            max_value=30,
            value=8
        )

        largeur_bande = st.slider(
            "Largeur de la zone bleue (pixels)",
            min_value=3,
            max_value=25,
            value=10
        )

        espace = st.slider(
            "Espace entre le câble et la zone bleue (pixels)",
            min_value=0,
            max_value=10,
            value=1
        )

        hz, wz = zone.shape[:2]

        centre_x = min(x + decalage, x + w - 1)

        bx1 = max(0, centre_x - largeur_bande // 2)
        bx2 = min(wz, bx1 + largeur_bande)

        by2 = max(0, y - espace)
        by1 = max(0, by2 - hauteur_bande)

        annotee = zone.copy()

        # OpenCV utilise BGR : vert puis bleu.
        cv2.rectangle(
            annotee,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 255, 0),
            1
        )

        bande_valide = bx2 > bx1 and by2 > by1

        if bande_valide:
            cv2.rectangle(
                annotee,
                (bx1, by1),
                (bx2 - 1, by2 - 1),
                (255, 0, 0),
                1
            )

        # Agrandissement sans lisser les pixels.
        zoom = cv2.resize(
            annotee,
            None,
            fx=6,
            fy=6,
            interpolation=cv2.INTER_NEAREST
        )

        with colonne2:
            st.image(
                cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB),
                caption="Vert : candidat câble — Bleu : zone observée"
            )

        if bande_valide:
            bande = zone[by1:by2, bx1:bx2]
            gris = cv2.cvtColor(bande, cv2.COLOR_BGR2GRAY)
            luminosite = float(gris.mean())

            st.metric(
                "Luminosité moyenne dans la zone bleue",
                f"{luminosite:.1f} / 255"
            )

            st.caption(
                "Cette valeur est un diagnostic, pas une décision "
                "ouvert/fermé. Un reflet ou un mauvais placement "
                "peut aussi la faire varier."
            )
        else:
            st.warning(
                "Pas assez de place au-dessus du câble. "
                "Vérifie la zone du mécanisme."
            )

        st.write(
            "Vérifie que le rectangle vert entoure le câble "
            "et que le rectangle bleu couvre la partie "
            "sombre ou claire juste au-dessus."
        )

    with st.expander("Voir les pixels jaunes détectés"):
        st.image(
            masque,
            caption="Blanc = pixels considérés comme jaunes",
            width=360,
            clamp=True
        )

except Exception as erreur:
    st.error(f"Erreur : {erreur}")

finally:
    if capture is not None:
        capture.release()

    if chemin_temp is not None:
        try:
            Path(chemin_temp).unlink(missing_ok=True)
        except OSError:
            st.warning("La copie temporaire n'a pas pu être supprimée.")