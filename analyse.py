import csv
import io
import itertools

import cv2
import numpy as np
import streamlit as st



def pretraiter_image(image):
    """
    Prétraitement robuste aux variations de lumière et aux ombres.

    1) niveaux de gris ;
    2) correction locale d'éclairage par division avec une image floutée ;
    3) normalisation ;
    4) CLAHE ;
    5) léger flou anti-bruit.

    Une ombre peut assombrir une partie de la ROI sans modifier la forme
    réelle de la lame. La correction locale réduit cet effet avant la
    comparaison OUVERT / FERME.
    """
    if image is None or image.size == 0:
        raise ValueError("Image vide.")

    if len(image.shape) == 3:
        gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gris = image.copy()

    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    # Estimation lente de l'éclairage local.
    illumination = cv2.GaussianBlur(
        gris,
        (0, 0),
        sigmaX=15,
        sigmaY=15
    )

    # Évite une division par une valeur nulle dans les zones très sombres.
    illumination = cv2.add(
        illumination,
        np.ones_like(illumination, dtype=np.uint8)
    )

    # "Aplatit" les différences locales de lumière / ombre.
    corrigee = cv2.divide(
        gris,
        illumination,
        scale=128
    )

    corrigee = cv2.normalize(
        corrigee,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    contraste = clahe.apply(corrigee)

    contraste = cv2.GaussianBlur(
        contraste,
        (3, 3),
        0
    )

    return contraste


def extraire_signature_structure(image):
    """
    Signature de forme utilisée uniquement pour distinguer :
    - lame visible normalement ;
    - main / objet qui masque réellement la ROI.

    On utilise le gradient de l'image déjà corrigée de l'éclairage.
    Une simple ombre change peu cette structure, alors qu'une main qui
    masque la lame change fortement les formes présentes.
    """
    traitee = pretraiter_image(image)

    gx = cv2.Sobel(
        traitee,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )
    gy = cv2.Sobel(
        traitee,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(gx, gy)

    signature = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return signature


def similarite_structure(image_a, image_b):
    """
    Similarité cosinus entre deux signatures structurelles.
    1.0 = très similaire ; 0.0 = très différente.
    """
    if image_a.shape != image_b.shape:
        return 0.0

    a = image_a.astype(np.float32).ravel()
    b = image_b.astype(np.float32).ravel()

    denominateur = float(
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominateur <= 1e-6:
        return 0.0

    return float(np.dot(a, b) / denominateur)


def preparer_references(references):
    """
    Préparer toutes les références avec le même traitement robuste
    à l'éclairage que celui appliqué aux images en temps réel.
    """
    preparees = {}

    for etat in ("ouvert", "ferme"):
        exemples = references.get(etat, [])

        if not exemples:
            raise ValueError(
                f"Il manque une référence pour l'état : {etat}."
            )

        preparees[etat] = [
            pretraiter_image(ref["image"])
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


def estimer_etat(image, preparees, marge, seuil_max):
    """
    Comparer l'image actuelle aux références OUVERT / FERMÉ après
    normalisation de luminosité + CLAHE.
    """
    traitee = pretraiter_image(image)

    for exemples in preparees.values():
        if any(ref.shape != traitee.shape for ref in exemples):
            raise ValueError(
                "La taille de la zone ne correspond pas aux références."
            )

    diff_ouverte = min(
        float(cv2.absdiff(traitee, ref).mean())
        for ref in preparees["ouvert"]
    )

    diff_fermee = min(
        float(cv2.absdiff(traitee, ref).mean())
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



def calibrer_references_automatiques(
    historique,
    nb_refs=2,
    nb_images_ouvert_initial=18
):
    """
    Calibration automatique V5 - basée sur les pics de STRUCTURE MECANIQUE.

    Pourquoi :
    - avec un câble long, le câble peut bouger longtemps dans la ROI ;
    - la V4 attendait un retour sous un seuil OUVERT et pouvait donc trouver
      0 événement, ou au contraire découper un même mouvement en plusieurs ;
    - ici on ne cherche plus des "épisodes" complets par seuil.

    Méthode :
    1) apprendre OUVERT au début ;
    2) ignorer au maximum la bande centrale où passe le câble ;
    3) calculer un score de changement de la MECANIQUE (bords/forme de lame) ;
    4) chercher les pics locaux les plus nets ;
    5) choisir 2 pics FERME cohérents entre eux.
    """
    if not historique:
        raise ValueError("Aucune image disponible pour la calibration.")

    images = [
        (int(numero), image.copy())
        for numero, image in historique
        if image is not None and image.size
    ]

    if len(images) < 30:
        raise ValueError(
            "Calibration trop courte. Gardez OUVERT environ 1 seconde, "
            "puis faites plusieurs fermetures complètes."
        )

    nb_base = min(
        max(12, int(nb_images_ouvert_initial)),
        max(12, len(images) // 5)
    )

    # ----------------------------------------------------------
    # Signature mécanique : contours de la lame.
    # On neutralise la bande centrale où le câble change de couleur/diamètre.
    # ----------------------------------------------------------
    def signature_mecanique(image):
        traitee = pretraiter_image(image)

        gx = cv2.Sobel(
            traitee,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )
        gy = cv2.Sobel(
            traitee,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(gx, gy)

        if float(magnitude.max()) > 0:
            magnitude = cv2.normalize(
                magnitude,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

        h, w = magnitude.shape

        masque = np.ones(
            (h, w),
            dtype=np.float32
        )

        # Le câble traverse principalement la zone horizontale centrale.
        # On la réduit fortement pour que la couleur/épaisseur influence moins.
        y1 = int(h * 0.34)
        y2 = int(h * 0.66)
        masque[y1:y2, :] = 0.20

        # On évite aussi les tout premiers pixels de bord.
        marge_y = max(1, int(h * 0.03))
        marge_x = max(1, int(w * 0.03))
        masque[:marge_y, :] = 0.0
        masque[-marge_y:, :] = 0.0
        masque[:, :marge_x] = 0.0
        masque[:, -marge_x:] = 0.0

        return magnitude.astype(np.float32) * masque

    signatures = [
        signature_mecanique(image)
        for _, image in images
    ]

    # ----------------------------------------------------------
    # 1) Référence OUVERT = médiane des premières images.
    # ----------------------------------------------------------
    pile_ouvert = np.stack(
        signatures[:nb_base],
        axis=0
    )

    ref_ouvert = np.median(
        pile_ouvert,
        axis=0
    ).astype(np.float32)

    scores = np.array(
        [
            float(np.mean(np.abs(sig - ref_ouvert)))
            for sig in signatures
        ],
        dtype=np.float32
    )

    # Lissage 3 images pour limiter les faux pics image-par-image.
    scores_lisses = scores.copy()

    if len(scores) >= 3:
        for i in range(1, len(scores) - 1):
            scores_lisses[i] = float(
                np.median(scores[i - 1:i + 2])
            )

    base = scores_lisses[:nb_base]
    med = float(np.median(base))
    mad = float(
        np.median(
            np.abs(base - med)
        )
    )

    # Seuil uniquement pour éliminer le bruit OUVERT.
    seuil_pic = med + max(
        2.0,
        4.0 * mad
    )

    # ----------------------------------------------------------
    # 2) Pics locaux candidats.
    # ----------------------------------------------------------
    pics_locaux = []

    for i in range(nb_base + 2, len(images) - 2):
        valeur = float(scores_lisses[i])

        if valeur < seuil_pic:
            continue

        voisinage = scores_lisses[i - 2:i + 3]

        if valeur >= float(np.max(voisinage)):
            pics_locaux.append(i)

    if not pics_locaux:
        raise ValueError(
            "Aucun pic mécanique détecté. Refaites plusieurs fermetures "
            "complètes à vitesse normale."
        )

    # Non-maximum suppression :
    # empêche 2-3 images adjacentes de la même fermeture d'être prises
    # comme plusieurs références.
    ordre = sorted(
        pics_locaux,
        key=lambda i: float(scores_lisses[i]),
        reverse=True
    )

    pics_filtres = []

    for i in ordre:
        if all(abs(i - j) >= 3 for j in pics_filtres):
            pics_filtres.append(i)

        if len(pics_filtres) >= 12:
            break

    if len(pics_filtres) < nb_refs:
        raise ValueError(
            f"Seulement {len(pics_filtres)} pic(s) mécanique(s) net(s) "
            "détecté(s). Refaites 4 à 6 fermetures complètes."
        )

    # ----------------------------------------------------------
    # 3) Choisir les 3 FERME les plus cohérents entre eux.
    # ----------------------------------------------------------
    def sim_cos(a, b):
        aa = a.astype(np.float32).ravel()
        bb = b.astype(np.float32).ravel()

        den = float(
            np.linalg.norm(aa) * np.linalg.norm(bb)
        )

        if den <= 1e-6:
            return 0.0

        return float(
            np.dot(aa, bb) / den
        )

    candidats = pics_filtres[:10]

    meilleure_combinaison = None
    meilleur_score = -1e9

    score_max = max(
        float(scores_lisses[i])
        for i in candidats
    )

    for combinaison in itertools.combinations(candidats, nb_refs):
        sims = []

        for a in range(nb_refs):
            for b in range(a + 1, nb_refs):
                sims.append(
                    sim_cos(
                        signatures[combinaison[a]],
                        signatures[combinaison[b]]
                    )
                )

        coherence = float(np.mean(sims))

        force = float(
            np.mean(
                [
                    float(scores_lisses[i]) / max(score_max, 1e-6)
                    for i in combinaison
                ]
            )
        )

        # Cohérence dominante, force du pic en second.
        score_global = (
            2.5 * coherence
            + 0.8 * force
        )

        if score_global > meilleur_score:
            meilleur_score = score_global
            meilleure_combinaison = combinaison

    selection_ferme = list(meilleure_combinaison)

    # ----------------------------------------------------------
    # 4) Choisir 3 OUVERT les plus stables dans la phase initiale.
    # ----------------------------------------------------------
    candidats_ouvert = sorted(
        range(nb_base),
        key=lambda i: float(scores_lisses[i])
    )

    selection_ouvert = []

    for i in candidats_ouvert:
        if all(abs(i - j) >= 2 for j in selection_ouvert):
            selection_ouvert.append(i)

        if len(selection_ouvert) >= nb_refs:
            break

    if len(selection_ouvert) < nb_refs:
        for i in candidats_ouvert:
            if i not in selection_ouvert:
                selection_ouvert.append(i)

            if len(selection_ouvert) >= nb_refs:
                break

    references = {
        "ouvert": [
            {
                "numero": images[i][0],
                "image": images[i][1].copy()
            }
            for i in sorted(selection_ouvert[:nb_refs])
        ],
        "ferme": [
            {
                "numero": images[i][0],
                "image": images[i][1].copy()
            }
            for i in sorted(selection_ferme[:nb_refs])
        ],
    }

    # ----------------------------------------------------------
    # 5) Contrôles finaux.
    # ----------------------------------------------------------
    preparees = preparer_references(references)

    distances_croisees = [
        float(cv2.absdiff(a, b).mean())
        for a in preparees["ouvert"]
        for b in preparees["ferme"]
    ]

    separation_min = min(distances_croisees)

    if separation_min < 3.0:
        raise ValueError(
            "Calibration ambiguë : OUVERT et FERME restent trop proches."
        )

    sig_ferme = [
        signatures[i]
        for i in selection_ferme
    ]

    sim_ff = []

    for a in range(len(sig_ferme)):
        for b in range(a + 1, len(sig_ferme)):
            sim_ff.append(
                sim_cos(
                    sig_ferme[a],
                    sig_ferme[b]
                )
            )

    coherence_ferme = float(
        np.mean(sim_ff)
    )

    # Similarité des FERME avec la référence OUVERT.
    sim_fo = [
        sim_cos(
            signatures[i],
            ref_ouvert
        )
        for i in selection_ferme
    ]

    proximite_ouvert = float(
        np.mean(sim_fo)
    )

    # Un vrai groupe FERME doit être plus cohérent entre lui-même
    # qu'avec OUVERT.
    if coherence_ferme <= proximite_ouvert + 0.02:
        raise ValueError(
            "Les pics détectés ressemblent encore trop à OUVERT. "
            "Refaites quelques fermetures complètes."
        )

    rapport = {
        "images_utilisees": len(images),
        "images_phase_ouvert": nb_base,
        "seuil_detection_ferme": round(float(seuil_pic), 2),
        "seuil_structure_ferme": "",
        "episodes_fermeture_detectes": len(pics_filtres),
        "pics_mecaniques_detectes": len(pics_filtres),
        "candidats_mecaniques_retenus": len(candidats),
        "separation_min_ouvert_ferme": round(separation_min, 2),
        "coherence_ferme": round(coherence_ferme, 3),
        "proximite_ferme_ouvert": round(proximite_ouvert, 3),
        "numeros_ouvert": [
            ref["numero"]
            for ref in references["ouvert"]
        ],
        "numeros_ferme": [
            ref["numero"]
            for ref in references["ferme"]
        ],
    }

    return references, rapport


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