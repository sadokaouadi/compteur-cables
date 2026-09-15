import hashlib
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from analyse import (
    afficher_analyse,
    estimer_etat,
    preparer_references,
)


st.set_page_config(
    page_title="Compteur de câbles",
    layout="wide",
)

st.title("Compteur de câbles — STARZ")
mode = st.radio("Source", ["Vidéo", "Caméra en direct"], horizontal=True)
if mode == "Caméra en direct":
    from camera_direct import afficher_camera
    afficher_camera()
    st.stop()
cam = st.session_state.pop("camera_direct", None)
if cam is not None:
    cam.fermer()

st.info(
    "Ajoute plusieurs exemples ouverts et fermés, "
    "puis analyse la vidéo."
)

video = st.file_uploader(
    "Choisir une vidéo",
    type=["mp4", "mov", "avi"],
)

if video is None:
    # Ne pas conserver les références d'une vidéo retirée.
    for cle in (
        "contexte_multi",
        "references_multi",
        "signature_analyse_multi",
        "resultat_analyse_multi",
    ):
        st.session_state.pop(cle, None)

    st.info("Charge une vidéo pour commencer.")
    st.stop()

with st.expander("Voir la vidéo originale"):
    st.video(video)

chemin_temp = None
capture = None

try:
    contenu = video.getvalue()
    identifiant_video = hashlib.sha256(contenu).hexdigest()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(video.name).suffix,
    ) as fichier_temp:
        fichier_temp.write(contenu)
        chemin_temp = fichier_temp.name

    capture = cv2.VideoCapture(chemin_temp)

    if not capture.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo.")

    nombre_images = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if nombre_images <= 0:
        raise ValueError(
            "Impossible de déterminer le nombre d'images."
        )

    numero_image = int(st.number_input(
        "Numéro de l'image à examiner",
        min_value=0,
        max_value=nombre_images - 1,
        value=0,
        step=1,
        key=f"image_multi_{identifiant_video}",
    ))

    capture.set(cv2.CAP_PROP_POS_FRAMES, numero_image)
    succes, image = capture.read()

    if not succes:
        raise ValueError("Impossible de lire cette image.")

    if fps > 0:
        st.caption(
            f"Position approximative : {numero_image / fps:.2f} s"
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hauteur, largeur = image.shape[:2]

    st.write(f"Dimensions : {largeur} × {hauteur} pixels")
    st.subheader("Zone du mécanisme")

    gauche, droite = st.slider(
        "Limites horizontales : gauche / droite (%)",
        min_value=0,
        max_value=100,
        value=(29, 38),
        key=f"zone_x_multi_{identifiant_video}",
    )

    haut, bas = st.slider(
        "Limites verticales : haut / bas (%)",
        min_value=0,
        max_value=100,
        value=(48, 55),
        key=f"zone_y_multi_{identifiant_video}",
    )

    x1 = int(largeur * gauche / 100)
    x2 = int(largeur * droite / 100)
    y1 = int(hauteur * haut / 100)
    y2 = int(hauteur * bas / 100)

    contexte = (identifiant_video, x1, y1, x2, y2)

    if st.session_state.get("contexte_multi") != contexte:
        st.session_state["contexte_multi"] = contexte
        st.session_state["references_multi"] = {
            "ouvert": [],
            "ferme": [],
        }
        st.session_state.pop("resultat_analyse_multi", None)

    if x2 <= x1 or y2 <= y1:
        st.warning("Choisis une zone non vide.")
        st.stop()

    zone = image_rgb[y1:y2, x1:x2]
    apercu = image_rgb.copy()

    cv2.rectangle(
        apercu,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 0),
        2,
    )

    colonne_image, colonne_zone = st.columns(2)

    with colonne_image:
        st.image(
            apercu,
            caption=f"Image n° {numero_image}",
            width=400,
        )

    with colonne_zone:
        st.image(
            zone,
            caption="Zone du mécanisme",
            width=300,
        )

    references = st.session_state["references_multi"]

    st.subheader("Références du mécanisme")
    st.write(
        "Chaque clic ajoute un exemple sans remplacer les anciens. "
        "Choisis des états clairement visibles, à différents "
        "moments de la vidéo."
    )
    st.caption(
        "Les numéros d'images dépendent de la vidéo. "
        "Changer la vidéo ou la zone efface les références. "
        "Elles restent uniquement en mémoire pendant la session."
    )

    colonnes = st.columns(2)

    for colonne, etat, libelle in zip(
        colonnes,
        ("ouvert", "ferme"),
        ("ouvert", "fermé"),
    ):
        with colonne:
            if st.button(
                f"Ajouter une référence {libelle}",
                key=f"ajouter_multi_{etat}",
            ):
                autre_etat = (
                    "ferme" if etat == "ouvert" else "ouvert"
                )

                deja_presente = any(
                    ref["numero"] == numero_image
                    for ref in references[etat]
                )

                conflit = any(
                    ref["numero"] == numero_image
                    for ref in references[autre_etat]
                )

                if deja_presente:
                    st.warning("Cette image est déjà enregistrée.")
                elif conflit:
                    st.warning(
                        "Cette image appartient déjà à l'autre état. "
                        "Supprime cette référence avant de la reclasser."
                    )
                else:
                    references[etat].append({
                        "image": zone.copy(),
                        "numero": numero_image,
                    })
                    st.session_state.pop(
                        "resultat_analyse_multi", None
                    )
                    st.success(
                        f"Référence {libelle} ajoutée : "
                        f"image {numero_image}."
                    )

    # Afficher toutes les références, avec suppression individuelle.
    colonnes = st.columns(2)

    for colonne, etat, libelle in zip(
        colonnes,
        ("ouvert", "ferme"),
        ("ouvert", "fermé"),
    ):
        with colonne:
            st.write(
                f"État {libelle} : "
                f"{len(references[etat])} référence(s)"
            )

            for ref in list(references[etat]):
                st.image(
                    ref["image"],
                    caption=(
                        f"Référence {libelle} — "
                        f"image {ref['numero']}"
                    ),
                    width=220,
                )

                if st.button(
                    f"Supprimer l'image {ref['numero']}",
                    key=f"supprimer_multi_{etat}_{ref['numero']}",
                ):
                    references[etat] = [
                        exemple
                        for exemple in references[etat]
                        if exemple["numero"] != ref["numero"]
                    ]
                    st.session_state.pop(
                        "resultat_analyse_multi", None
                    )
                    st.rerun()

    if not references["ouvert"] or not references["ferme"]:
        st.info(
            "Ajoute au moins une référence ouverte "
            "et une référence fermée."
        )
        st.stop()

    preparees = preparer_references(references)

    # Éviter deux exemples visuellement identiques de classes opposées.
    conflit_visuel = any(
        cv2.absdiff(ouverte, fermee).max() == 0
        for ouverte in preparees["ouvert"]
        for fermee in preparees["ferme"]
    )

    if conflit_visuel:
        st.warning(
            "Une référence ouverte et une référence fermée "
            "sont identiques. Supprime l'exemple mal classé."
        )
        st.stop()

    st.subheader("État estimé du mécanisme")

    marge = st.slider(
        "Marge minimale entre les deux différences",
        min_value=0.0,
        max_value=20.0,
        value=0.5,
        step=0.1,
        key="marge_multi",
    )

    seuil_max = st.slider(
        "Différence maximale acceptée",
        min_value=1.0,
        max_value=100.0,
        value=39.0,
        step=1.0,
        key="seuil_multi",
    )

    confirmation = int(st.number_input(
        "Images consécutives pour confirmer un état",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="confirmation_multi",
    ))

    st.caption(
        "Une confirmation sur une seule image peut détecter "
        "des mouvements courts, mais aussi créer de faux événements."
    )

    actuelle = cv2.cvtColor(zone, cv2.COLOR_RGB2GRAY)

    etat, diff_ouverte, diff_fermee = estimer_etat(
        actuelle,
        preparees,
        marge,
        seuil_max,
    )

    if etat == "OUVERT":
        st.success("État estimé : OUVERT")
    elif etat == "FERME":
        st.info("État estimé : FERMÉ")
    else:
        st.warning("État estimé : INDÉTERMINÉ")

    st.write(
        f"Différence avec la référence ouverte "
        f"la plus proche : {diff_ouverte:.2f}"
    )
    st.write(
        f"Différence avec la référence fermée "
        f"la plus proche : {diff_fermee:.2f}"
    )

    st.caption(
        "Ces différences de pixels ne sont pas des pourcentages "
        "de confiance. Une image comparée à elle-même donne zéro."
    )

    afficher_analyse(
        chemin_temp,
        references,
        (x1, y1, x2, y2),
        marge,
        seuil_max,
        confirmation,
        contexte,
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
            st.warning(
                "La copie temporaire n'a pas pu être supprimée."
            )