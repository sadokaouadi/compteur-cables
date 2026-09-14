import csv
import io

import cv2
import streamlit as st


def preparer_references(references):
    """Convertir toutes les références RGB en niveaux de gris."""
    preparees = {}

    for etat in ("ouvert", "ferme"):
        exemples = references.get(etat, [])

        if not exemples:
            raise ValueError(
                f"Il manque une référence pour l'état : {etat}."
            )

        preparees[etat] = [
            cv2.cvtColor(ref["image"], cv2.COLOR_RGB2GRAY)
            for ref in exemples
        ]

    formes = {
        image.shape
        for exemples in preparees.values()
        for image in exemples
    }

    if len(formes) != 1:
        raise ValueError(
            "Les références doivent avoir la même taille."
        )

    return preparees


def estimer_etat(gris, preparees, marge, seuil_max):
    """Comparer l'image à l'exemple le plus proche de chaque état."""
    for exemples in preparees.values():
        if any(ref.shape != gris.shape for ref in exemples):
            raise ValueError(
                "La taille de la zone ne correspond pas aux références."
            )

    diff_ouverte = min(
        float(cv2.absdiff(gris, ref).mean())
        for ref in preparees["ouvert"]
    )

    diff_fermee = min(
        float(cv2.absdiff(gris, ref).mean())
        for ref in preparees["ferme"]
    )

    ecart = abs(diff_ouverte - diff_fermee)

    if (
        ecart <= marge
        or min(diff_ouverte, diff_fermee) > seuil_max
    ):
        etat = "INDETERMINE"
    elif diff_ouverte < diff_fermee:
        etat = "OUVERT"
    else:
        etat = "FERME"

    return etat, diff_ouverte, diff_fermee


def afficher_analyse(
    chemin_video,
    references,
    limites,
    marge,
    seuil_max,
    confirmation,
    contexte
):
    st.subheader("Analyse automatique : vidéo complète")

    # Un résultat n'est réutilisé que pour la même configuration.
    signature = (
        contexte,
        float(marge),
        float(seuil_max),
        int(confirmation),
        tuple(ref["numero"] for ref in references["ouvert"]),
        tuple(ref["numero"] for ref in references["ferme"]),
    )

    if st.session_state.get("signature_analyse_multi") != signature:
        st.session_state["signature_analyse_multi"] = signature
        st.session_state.pop("resultat_analyse_multi", None)

    if st.button("Analyser toute la vidéo"):
        st.session_state.pop("resultat_analyse_multi", None)

        capture = None

        try:
            preparees = preparer_references(references)
            x1, y1, x2, y2 = limites

            capture = cv2.VideoCapture(chemin_video)

            if not capture.isOpened():
                raise ValueError("Impossible d'ouvrir la vidéo.")

            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if total <= 0:
                raise ValueError(
                    "Impossible de déterminer le nombre d'images."
                )

            etat_stable = None
            candidat = None
            repetitions = 0
            debut_candidat = None
            debut_fermeture = None
            fermeture_initiale = False

            evenements = []
            journal = []

            with st.spinner("Analyse de la vidéo en cours..."):
                for numero in range(total):
                    succes, image = capture.read()

                    if not succes:
                        raise ValueError(
                            f"Lecture interrompue à l'image {numero} "
                            f"sur {total}. Aucun résultat complet "
                            "n'est présenté."
                        )

                    hauteur, largeur = image.shape[:2]

                    if not (
                        0 <= x1 < x2 <= largeur
                        and 0 <= y1 < y2 <= hauteur
                    ):
                        raise ValueError(
                            "La zone sélectionnée est invalide."
                        )

                    zone = image[y1:y2, x1:x2]
                    gris = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

                    etat, diff_ouverte, diff_fermee = estimer_etat(
                        gris, preparees, marge, seuil_max
                    )

                    if etat == "INDETERMINE":
                        candidat = None
                        repetitions = 0
                        debut_candidat = None

                    else:
                        if etat == candidat:
                            repetitions += 1
                        else:
                            candidat = etat
                            repetitions = 1
                            debut_candidat = numero

                        if (
                            repetitions >= confirmation
                            and etat != etat_stable
                        ):
                            precedent = etat_stable
                            etat_stable = etat

                            if etat == "FERME":
                                if precedent == "OUVERT":
                                    debut_fermeture = debut_candidat
                                else:
                                    # Pas d'ouverture confirmée avant.
                                    debut_fermeture = None
                                    fermeture_initiale = True

                            elif (
                                precedent == "FERME"
                                and debut_fermeture is not None
                            ):
                                evenements.append({
                                    "Fermeture": len(evenements) + 1,
                                    "Début fermé estimé": debut_fermeture,
                                    "Début réouverture estimé": (
                                        debut_candidat
                                    ),
                                    "Réouverture confirmée à": numero,
                                })
                                debut_fermeture = None

                    journal.append({
                        "Image": numero,
                        "État estimé": etat,
                        "État confirmé": (
                            etat_stable or "NON INITIALISE"
                        ),
                        "Différence ouvert": round(diff_ouverte, 2),
                        "Différence fermé": round(diff_fermee, 2),
                    })

            st.session_state["resultat_analyse_multi"] = {
                "evenements": evenements,
                "journal": journal,
                "total": total,
                "fermeture_initiale": fermeture_initiale,
                "fermeture_incomplete": debut_fermeture,
            }

        except Exception as erreur:
            st.error(f"Analyse impossible : {erreur}")

        finally:
            if capture is not None:
                capture.release()

    resultat = st.session_state.get("resultat_analyse_multi")

    if resultat is None:
        return

    evenements = resultat["evenements"]
    journal = resultat["journal"]

    st.caption(
        f"{resultat['total']} images analysées, "
        f"de 0 à {resultat['total'] - 1}."
    )

    nombre_fermetures = len(evenements)
    nombre_cables, reste = divmod(nombre_fermetures, 3)

    st.metric("Fermetures complètes détectées", nombre_fermetures)
    st.metric("Câbles estimés — groupes de 3 fermetures", nombre_cables)

    st.caption(
        "Cette estimation suppose trois fermetures par câble. "
        "Une fermeture manquée ou un faux événement peut fausser "
        "le total et le regroupement."
    )

    if reste:
        st.warning(
            f"{reste} fermeture(s) restent sans groupe complet."
        )

    if resultat["fermeture_initiale"]:
        st.warning(
            "Le premier état confirmé était fermé. "
            "Cette fermeture sans ouverture précédente "
            "n'a pas été comptée."
        )

    if resultat["fermeture_incomplete"] is not None:
        st.warning(
            "Une fermeture commencée à l'image "
            f"{resultat['fermeture_incomplete']} "
            "n'a pas de réouverture confirmée avant la fin."
        )

    # CSV des groupes de trois fermetures.
    colonnes = ["Cable"]

    for numero in range(1, 4):
        colonnes.extend([
            f"Fermeture_{numero}_debut",
            f"Fermeture_{numero}_reouverture",
        ])

    fichier_csv = io.StringIO()
    ecrivain = csv.DictWriter(fichier_csv, fieldnames=colonnes)
    ecrivain.writeheader()

    for index in range(nombre_cables):
        groupe = evenements[index * 3:index * 3 + 3]
        ligne = {"Cable": index + 1}

        for numero, evenement in enumerate(groupe, start=1):
            ligne[f"Fermeture_{numero}_debut"] = (
                evenement["Début fermé estimé"]
            )
            ligne[f"Fermeture_{numero}_reouverture"] = (
                evenement["Début réouverture estimé"]
            )

        ecrivain.writerow(ligne)

    st.download_button(
        "Télécharger le résultat CSV",
        data=fichier_csv.getvalue().encode("utf-8-sig"),
        file_name="resultat_comptage_cables.csv",
        mime="text/csv",
    )

    st.subheader("Détail des fermetures")

    if not evenements:
        st.warning("Aucune fermeture complète détectée.")

    for evenement in evenements:
        st.text(
            f"Fermeture {evenement['Fermeture']} : "
            f"début fermé = {evenement['Début fermé estimé']}, "
            f"début réouverture = "
            f"{evenement['Début réouverture estimé']}, "
            f"réouverture confirmée à = "
            f"{evenement['Réouverture confirmée à']}"
        )

    with st.expander("Voir le détail des images"):
        for ligne in journal:
            st.text(
                f"Image {ligne['Image']} : "
                f"{ligne['État estimé']} | "
                f"confirmé : {ligne['État confirmé']} | "
                f"diff. ouvert : {ligne['Différence ouvert']} | "
                f"diff. fermé : {ligne['Différence fermé']}"
            )