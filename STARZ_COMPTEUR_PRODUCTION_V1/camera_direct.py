"""Caméra locale : acquisition, lecture OF par CODE_128 et comptage continu."""
import csv
import io
import platform
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components
import zxingcpp

from analyse import (
    preparer_references,
    estimer_etat,
    extraire_signature_structure,
    similarite_structure,
    calibrer_references_automatiques,
)


# ============================================================
# CODE-BARRES / OF
# ============================================================

# Format actuellement attendu dans vos feuilles :
# 48247-1, 48246-1, etc.
FORMAT_OF = re.compile(r"^\d{5}-\d$")

# ============================================================
# V7 — COMPTEUR GLOBAL AUTOMATIQUE
# ============================================================
AUTO_CALIBRATION_CIBLE_IMAGES = 800
AUTO_CALIBRATION_NB_REFS = 2


def detecter_code_barres(image):
    """
    Recherche un code-barres valide dans l'image entière.

    Retour :
        (texte, format) si un OF valide est trouvé,
        (None, None) sinon.
    """
    try:
        resultats = zxingcpp.read_barcodes(image)

        for resultat in resultats:
            texte = (resultat.text or "").strip()

            if FORMAT_OF.fullmatch(texte):
                return texte, str(resultat.format)

    except Exception as exc:
        print("Erreur lecture code-barres :", exc)

    return None, None


def normaliser_work_order_scanner(texte):
    """
    Production V1.7 — correction automatique du scanner USB.

    Cas gérés :
    1) Le scanner renvoie déjà le bon code (ex. 48414-2)
       -> on conserve exactement la valeur.
    2) Le scanner est en disposition US alors que Windows est en
       clavier français AZERTY et les chiffres deviennent des symboles
       (ex. ' _ & ( ) é ...)
       -> conversion automatique vers les chiffres réels.

    La conversion n'est appliquée que lorsque le scan ne contient
    aucun chiffre ASCII et ressemble clairement à la rangée numérique
    d'un clavier AZERTY. Cela évite de modifier un code déjà correct.
    """
    brut = (
        (texte or "")
        .replace("\r", "")
        .replace("\n", "")
        .strip()
    )

    if not brut:
        return None

    if len(brut) > 100:
        return None

    # Si le scanner retourne déjà des chiffres, on considère le scan
    # comme correct et on ne touche à rien.
    if any(car.isdigit() for car in brut):
        return brut

    conversion_azerty = {
        "&": "1",
        "é": "2",
        '"': "3",
        "'": "4",
        "(": "5",
        "-": "6",
        "è": "7",
        "_": "8",
        "ç": "9",
        "à": "0",
        ")": "-",
    }

    # Caractères typiques produits par les touches numériques quand
    # un scanner configuré US est interprété par Windows FR/AZERTY.
    caracteres_scanner_azerty = set(conversion_azerty.keys())

    # Ignorer les espaces parasites éventuels envoyés par certains
    # lecteurs avant/après le code.
    compact = brut.replace(" ", "")

    # Conversion uniquement si le code est composé entièrement de
    # caractères provenant de cette rangée clavier et contient au
    # moins 2 caractères : cas typique d'un WO numérique/hyphen.
    if (
        len(compact) >= 2
        and all(
            car in caracteres_scanner_azerty
            for car in compact
        )
    ):
        corrige = "".join(
            conversion_azerty[car]
            for car in compact
        )
        return corrige

    # Si le scan contient au moins un caractère accentué très typique,
    # on corrige uniquement les caractères connus et on garde le reste.
    # Cela couvre quelques lecteurs qui mélangent caractères et suffixes.
    signes_forts = set("éèçà")

    if any(car in signes_forts for car in brut):
        return "".join(
            conversion_azerty.get(car, car)
            for car in brut
        )

    # Sinon, on garde le code tel qu'il a été reçu.
    return brut


# ============================================================
# COMPTEUR
# ============================================================

class Compteur:
    """
    Détecte les fermetures complètes OUVERT -> FERME -> OUVERT
    puis applique la logique réelle de la machine :

        F1 = traçage / dénudage        -> pas de câble
        F2 = coupe réelle du câble     -> +1 câble
        F3 = préparation / traçage suivant -> pas de câble

    Cas spéciaux gérés :
    - début d'une nouvelle commande : 2 fermetures d'initialisation
      (F2 + F3) peuvent être ignorées ;
    - pause longue : resynchronisation sur F1 à la reprise ;
    - arrêt après F1 + F2 : le câble est déjà compté au moment de F2 ;
    - fin de commande avec seulement 2 fermetures : le second événement
      peut encore représenter la coupe utile et le câble est compté.
    """

    def __init__(
        self,
        confirmation=1,
        ignorer_initiales=2,
        seuil_pause_s=5.0
    ):
        self.confirmation = confirmation

        # Détection OUVERT / FERME
        self.stable = None
        self.candidat = None
        self.debut = None
        self.repetitions = 0
        self.initiale = False

        # Journal des fermetures complètes
        self.evenements = []

        # Logique câble
        self.cables = 0
        self.phase = 0  # 0=F1 attendu, 1=F2 attendu, 2=F3 attendu
        self.ignorer_initiales = max(0, int(ignorer_initiales))
        self.initiales_restantes = self.ignorer_initiales
        self.seuil_pause_s = float(seuil_pause_s)

        self.derniere_fermeture_s = None
        self.pauses_detectees = 0
        self.fermetures_ignorees_initialisation = 0

        # Après une vraie pause au milieu d'un cycle, la machine réalise
        # 2 mouvements de remise en route qui NE correspondent pas à
        # un câble coupé. Ils doivent être ignorés.
        self.reprise_initiales_restantes = 0
        self.fermetures_ignorees_reprise = 0

        # V1.5 :
        # nombre de reprises où une occlusion + une longue attente
        # ont forcé la protection des 2 premiers mouvements.
        self.reprises_forcees_occlusion = 0

    def _classer_fermeture(self, evenement):
        """
        Classe une fermeture complète en INIT, F1, F2 ou F3.
        Le câble est incrémenté immédiatement à F2.
        """
        maintenant = evenement["Secondes_reouverture"]

        # ----------------------------------------------------
        # 1) Initialisation spéciale de nouvelle commande
        # ----------------------------------------------------
        if self.initiales_restantes > 0:
            numero_init = (
                self.ignorer_initiales
                - self.initiales_restantes
                + 1
            )

            evenement["Etape_machine"] = f"INIT_{numero_init}"
            evenement["Cable_compte"] = ""
            evenement["Pause_avant"] = "NON"

            self.initiales_restantes -= 1
            self.fermetures_ignorees_initialisation += 1
            self.derniere_fermeture_s = maintenant

            # Après les 2 fermetures initiales, le prochain vrai cycle
            # repart sur F1.
            if self.initiales_restantes == 0:
                self.phase = 0

            return

        # ----------------------------------------------------
        # 2) Détection d'une pause / reprise
        # ----------------------------------------------------
        # CAS MACHINE OBSERVÉ :
        #
        # avant pause :
        #     F1 -> F2 = câble réellement coupé (+1)
        #     puis la machine s'arrête avant F3
        #
        # à la reprise :
        #     2 mouvements de remise en route
        #     => aucun câble ne doit être ajouté
        #
        # L'ancienne logique remettait seulement phase=0 :
        # le 1er mouvement devenait F1 et le 2e F2 => faux +1 câble.
        #
        # La nouvelle logique ignore explicitement ces 2 mouvements.

        if self.reprise_initiales_restantes > 0:
            numero_reprise = 3 - self.reprise_initiales_restantes

            evenement["Etape_machine"] = (
                f"REPRISE_INIT_{numero_reprise}"
            )
            evenement["Cable_compte"] = ""
            evenement["Pause_avant"] = "NON"

            self.reprise_initiales_restantes -= 1
            self.fermetures_ignorees_reprise += 1
            self.derniere_fermeture_s = maintenant

            if self.reprise_initiales_restantes == 0:
                self.phase = 0

            return

        if self.derniere_fermeture_s is not None:
            ecart = maintenant - self.derniere_fermeture_s

            # Une longue attente alors que le cycle est incomplet
            # signifie une pause/reprise.
            if (
                ecart >= self.seuil_pause_s
                and self.phase != 0
            ):
                self.pauses_detectees += 1

                # Le mouvement courant est déjà le premier des
                # 2 mouvements de remise en route à ignorer.
                evenement["Etape_machine"] = "REPRISE_INIT_1"
                evenement["Cable_compte"] = ""
                evenement["Pause_avant"] = "OUI"

                self.fermetures_ignorees_reprise += 1
                self.reprise_initiales_restantes = 1
                self.phase = 0
                self.derniere_fermeture_s = maintenant
                return

        evenement["Pause_avant"] = "NON"

        # ----------------------------------------------------
        # 3) Machine à états F1 / F2 / F3
        # ----------------------------------------------------
        if self.phase == 0:
            evenement["Etape_machine"] = "F1_TRACAGE"
            evenement["Cable_compte"] = ""
            self.phase = 1

        elif self.phase == 1:
            evenement["Etape_machine"] = "F2_COUPE"
            self.cables += 1
            evenement["Cable_compte"] = self.cables
            self.phase = 2

        else:
            evenement["Etape_machine"] = "F3_PREPARATION_SUIVANT"
            evenement["Cable_compte"] = ""
            self.phase = 0

        self.derniere_fermeture_s = maintenant

    def ajouter(self, etat, numero, secondes):
        # INDETERMINE / OCCLUSION ne doivent jamais créer une fermeture.
        # On gèle la logique machine F1/F2/F3.
        if etat in ('INDETERMINE', 'OCCLUSION'):
            self.candidat = None
            self.repetitions = 0
            return

        if etat != self.candidat:
            self.candidat = etat
            self.repetitions = 0
            self.premier = (numero, secondes)

        self.repetitions += 1

        if self.repetitions < self.confirmation or etat == self.stable:
            return

        precedent, self.stable = self.stable, etat

        if etat == 'FERME':
            self.debut = self.premier if precedent == 'OUVERT' else None
            self.initiale |= precedent is None

        elif precedent == 'FERME' and self.debut is not None:
            evenement = {
                'Fermeture': len(self.evenements) + 1,
                'Image_debut': self.debut[0],
                'Secondes_debut': round(self.debut[1], 3),
                'Image_reouverture': self.premier[0],
                'Secondes_reouverture': round(self.premier[1], 3),
                'Image_confirmation': numero,
            }

            self._classer_fermeture(evenement)
            self.evenements.append(evenement)
            self.debut = None

    def reprendre_apres_occlusion(self, secondes=None):
        """
        Repart proprement depuis OUVERT sans créer de fausse fermeture.

        V1.5 :
        si l'occlusion a eu lieu pendant une vraie pause (longue attente
        depuis la dernière fermeture), on arme immédiatement la protection
        de reprise : les 2 premiers mouvements mécaniques après la pause
        sont ignorés.

        Cela corrige le cas où la main masque la ROI pendant la pause :
        auparavant la sortie d'occlusion remettait seulement l'état visuel
        sur OUVERT, mais la phase F1/F2/F3 pouvait ensuite compter le
        deuxième mouvement de reprise comme un vrai câble.
        """
        self.stable = 'OUVERT'
        self.candidat = None
        self.repetitions = 0
        self.debut = None

        if (
            secondes is not None
            and self.derniere_fermeture_s is not None
            and self.reprise_initiales_restantes == 0
        ):
            ecart = float(secondes) - float(self.derniere_fermeture_s)

            if ecart >= self.seuil_pause_s:
                self.pauses_detectees += 1
                self.reprise_initiales_restantes = 2
                self.phase = 0
                self.reprises_forcees_occlusion += 1



# ============================================================
# CAMERA
# ============================================================

class Camera:
    def __init__(
        self,
        index,
        limites,
        rotation="Aucune",
        activer_barcode=False,
        verrouiller_exposition=True
    ):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        self.limites = limites
        self.rotation = rotation
        self.derniere = None
        self.historique = deque(maxlen=1200)

        # ------------------------------------------------------
        # V6 — BUFFER DÉDIÉ À LA CALIBRATION
        # ------------------------------------------------------
        # Contrairement à l'historique d'aperçu, ce buffer conserve aussi
        # l'horodatage monotone de chaque ROI. Il sert ensuite à rejouer les
        # mouvements effectués pendant la calibration pour ne perdre aucun
        # câble produit avant le démarrage du comptage temps réel.
        self.calibration_buffer = deque(maxlen=2500)
        self.calibration_capture_active = False
        self.calibration_buffer_sature = False
        self.calibration_started_monotonic = None
        self.calibration_started_at = None

        self.images_rejouees_calibration = 0
        self.fermetures_recuperees_calibration = 0
        self.cables_recuperes_calibration = 0

        # Exposition caméra :
        # on laisse d'abord la caméra se stabiliser sous la lumière normale,
        # puis on tente de figer cette exposition pour éviter que l'auto-exposure
        # change brutalement pendant le comptage.
        self.verrouiller_exposition = bool(verrouiller_exposition)
        self.exposition_capturee = None
        self.exposition_reelle = None
        self.auto_exposure_reelle = None
        self.exposition_verrouillee = False

        self.numero = 0
        self.erreur = None
        self.actif = False

        self.compteur = Compteur()
        self.etat = 'NON INITIALISE'

        # -------------------------
        # PROTECTION MAIN / OCCLUSION
        # -------------------------
        self.protection_occlusion = True
        self.occlusion_active = False
        self.occlusion_candidat = 0
        self.retour_ouvert_candidat = 0
        self.occlusions_detectees = 0
        self.frames_occlusion = 0

        self.diff_ouverte = None
        self.diff_fermee = None
        self.fraction_changement = 0.0

        # Références brutes utilisées uniquement pour savoir si une très
        # grande partie de la ROI est masquée par une main / un objet.
        self.refs_brutes = None

        # Signatures de forme OUVERT / FERME.
        # Elles servent à reconnaître une vraie obstruction par une main
        # même lorsque la luminosité change.
        self.refs_structure = None
        self.seuil_structure_occlusion = 0.45
        self.similarite_structure_max = None

        # Validation structurelle séparée pour éviter qu'une main partielle
        # soit acceptée comme FERME avant d'être reconnue comme OCCLUSION.
        self.seuil_validation_ouvert = 0.60
        self.seuil_validation_ferme = 0.60
        self.similarite_ouverte = None
        self.similarite_fermee = None

        # FPS demandé au pilote et valeur déclarée par le pilote.
        self.fps_demande = 30.0
        self.fps_pilote = None

        # -------------------------
        # CODE-BARRES / OF
        # -------------------------
        self.code_barres = None
        self.code_barres_format = None

        # Lecture OF facultative pendant les tests machine.
        self.scan_barcode_actif = bool(activer_barcode)

        # OF verrouillé au moment où le comptage démarre
        self.of_compteur = None

        # -------------------------
        # TEST DE VALIDATION
        # -------------------------
        self.confirmation_test = None
        self.test_meta = {}
        self.test_started_at = None
        self.test_ended_at = None
        self.test_started_monotonic = None
        self.test_ended_monotonic = None

        self.heartbeat = time.monotonic()
        self.depart = self.heartbeat

        self.thread = threading.Thread(
            target=self._lire,
            args=(index,),
            daemon=True
        )
        self.thread.start()

    def _lire(self, index):
        cap = None

        try:
            backend = (
                cv2.CAP_DSHOW
                if platform.system() == 'Windows'
                else cv2.CAP_ANY
            )

            cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                raise ValueError(
                    'Caméra inaccessible : fermez Caméra Windows, '
                    'puis essayez un autre index.'
                )

            # Votre webcam a été validée en 1280 × 720.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, int(self.fps_demande))
            self.fps_pilote = cap.get(cv2.CAP_PROP_FPS)

            # ==================================================
            # VERROUILLAGE DE L'EXPOSITION
            # ==================================================
            # La caméra reste quelques images en réglage normal pour que
            # l'exposition se stabilise sous l'éclairage de production.
            # Ensuite, sous Windows/DirectShow, on demande le mode manuel
            # (0.25) et on réapplique la valeur d'exposition observée.
            #
            # Important : certains drivers ignorent CAP_PROP_AUTO_EXPOSURE.
            # On mémorise donc le résultat pour l'afficher dans l'interface.
            if self.verrouiller_exposition:
                for _ in range(20):
                    ok_warmup, _ = cap.read()
                    if not ok_warmup:
                        break

                exposition_avant = cap.get(cv2.CAP_PROP_EXPOSURE)
                self.exposition_capturee = exposition_avant

                if platform.system() == 'Windows':
                    manuel_demande = cap.set(
                        cv2.CAP_PROP_AUTO_EXPOSURE,
                        0.25
                    )
                else:
                    manuel_demande = cap.set(
                        cv2.CAP_PROP_AUTO_EXPOSURE,
                        0
                    )

                exposition_demandee = cap.set(
                    cv2.CAP_PROP_EXPOSURE,
                    exposition_avant
                )

                time.sleep(0.15)

                self.exposition_reelle = cap.get(
                    cv2.CAP_PROP_EXPOSURE
                )
                self.auto_exposure_reelle = cap.get(
                    cv2.CAP_PROP_AUTO_EXPOSURE
                )

                self.exposition_verrouillee = bool(
                    manuel_demande and exposition_demandee
                )

            compteur_scan_barcode = 0

            while not self.stop_event.is_set():

                # V1.5 — la caméra ne doit pas dépendre de l'onglet
                # navigateur pendant une production active.
                #
                # Les navigateurs peuvent ralentir/suspendre les rafraîchissements
                # Streamlit quand l'utilisateur change de fenêtre. L'ancien
                # timeout de 30 s arrêtait alors la caméra alors que le matériel
                # était toujours connecté.
                #
                # Pendant le comptage : aucun arrêt sur heartbeat UI.
                # Hors production : on garde seulement une sécurité très large
                # pour éviter une caméra orpheline pendant des heures.
                if (
                    not self.actif
                    and time.monotonic() - self.heartbeat > 1800
                ):
                    raise ValueError(
                        'Session inactive depuis 30 minutes : caméra arrêtée.'
                    )

                ok, image = cap.read()

                if not ok:
                    raise ValueError(
                        'Lecture caméra interrompue. '
                        'Le comptage est partiel.'
                    )

                # ==================================================
                # ORIENTATION DE LA CAMERA
                # La rotation est appliquée AVANT le code-barres et
                # AVANT le calcul de la ROI de comptage.
                # ==================================================
                if self.rotation == "90° droite":
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == "90° gauche":
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self.rotation == "180°":
                    image = cv2.rotate(image, cv2.ROTATE_180)

                compteur_scan_barcode += 1
                now = time.monotonic()

                # ==================================================
                # 1) LECTURE CODE-BARRES SUR L'IMAGE ENTIÈRE
                # ==================================================
                with self.lock:
                    scanner_barcode = (
                        self.scan_barcode_actif
                        and not self.actif
                    )

                # Une tentative toutes les 5 images suffit.
                if scanner_barcode and compteur_scan_barcode % 5 == 0:
                    code_detecte, format_detecte = detecter_code_barres(image)

                    if code_detecte:
                        with self.lock:
                            # Re-vérification car l'état peut avoir changé
                            # pendant le décodage.
                            if self.scan_barcode_actif and not self.actif:
                                self.code_barres = code_detecte
                                self.code_barres_format = format_detecte
                                self.scan_barcode_actif = False

                                print(
                                    "CODE-BARRES DETECTE :",
                                    code_detecte,
                                    "|",
                                    format_detecte
                                )

                # ==================================================
                # 2) ROI DE COMPTAGE
                # ==================================================
                h, w = image.shape[:2]

                g, d, t, b = self.limites

                x1 = int(w * g / 100)
                x2 = int(w * d / 100)
                y1 = int(h * t / 100)
                y2 = int(h * b / 100)

                zone_bgr = image[y1:y2, x1:x2]

                if not zone_bgr.size:
                    raise ValueError(
                        'Zone vide : modifiez les limites.'
                    )

                zone = cv2.cvtColor(
                    zone_bgr,
                    cv2.COLOR_BGR2RGB
                )

                with self.lock:
                    self.numero += 1
                    self.derniere = (
                        image,
                        zone,
                        (x1, y1, x2, y2)
                    )

                    self.historique.append(
                        (
                            self.numero,
                            zone.copy()
                        )
                    )

                    # V6 : pendant la calibration automatique, on conserve
                    # chaque ROI avec son vrai timestamp. Ce buffer continue
                    # à enregistrer même après la création des références,
                    # jusqu'au clic sur « Démarrer le comptage ».
                    if self.calibration_capture_active:
                        if (
                            self.calibration_buffer.maxlen is not None
                            and len(self.calibration_buffer)
                            >= self.calibration_buffer.maxlen
                        ):
                            self.calibration_buffer_sature = True

                        self.calibration_buffer.append(
                            (
                                self.numero,
                                zone.copy(),
                                now
                            )
                        )

                    # ==================================================
                    # 3) ANALYSE DU MOUVEMENT SI COMPTAGE ACTIF
                    # ==================================================
                    if self.actif:
                        gris = cv2.cvtColor(
                            zone,
                            cv2.COLOR_RGB2GRAY
                        )

                        etat_brut, diff_ouverte, diff_fermee = estimer_etat(
                            gris,
                            self.refs,
                            self.marge,
                            self.seuil
                        )

                        self.diff_ouverte = diff_ouverte
                        self.diff_fermee = diff_fermee

                        # --------------------------------------------------
                        # PROTECTION MAIN / OBJET DANS LA ROI
                        # --------------------------------------------------
                        # On ne se base plus uniquement sur la luminosité brute.
                        # Une ombre peut rendre la ROI beaucoup plus sombre sans
                        # masquer la lame. On compare donc aussi la STRUCTURE
                        # (gradient / formes) avec les références OUVERT / FERME.

                        fraction_min = 0.0

                        if self.refs_brutes:
                            fractions = []

                            for groupe in ('ouvert', 'ferme'):
                                for ref_brute in self.refs_brutes[groupe]:
                                    if ref_brute.shape == gris.shape:
                                        difference = cv2.absdiff(
                                            gris,
                                            ref_brute
                                        )
                                        fraction = float(
                                            (difference > 40).mean()
                                        )
                                        fractions.append(fraction)

                            if fractions:
                                fraction_min = min(fractions)

                        self.fraction_changement = fraction_min

                        signature_actuelle = extraire_signature_structure(
                            gris
                        )

                        similarites_ouvert = []
                        similarites_ferme = []

                        if self.refs_structure:
                            similarites_ouvert = [
                                similarite_structure(
                                    signature_actuelle,
                                    ref_structure
                                )
                                for ref_structure
                                in self.refs_structure["ouvert"]
                            ]

                            similarites_ferme = [
                                similarite_structure(
                                    signature_actuelle,
                                    ref_structure
                                )
                                for ref_structure
                                in self.refs_structure["ferme"]
                            ]

                        similarite_ouverte = (
                            max(similarites_ouvert)
                            if similarites_ouvert
                            else 0.0
                        )
                        similarite_fermee = (
                            max(similarites_ferme)
                            if similarites_ferme
                            else 0.0
                        )

                        similarite_max = max(
                            similarite_ouverte,
                            similarite_fermee
                        )

                        self.similarite_ouverte = similarite_ouverte
                        self.similarite_fermee = similarite_fermee
                        self.similarite_structure_max = similarite_max

                        # --------------------------------------------------
                        # VERSION V5 : logique simplifiée et plus robuste
                        # --------------------------------------------------
                        # Les seuils de similarité structurelle ne servent PLUS
                        # à valider chaque OUVERT / FERME.
                        #
                        # Pourquoi ?
                        # Une vraie lame peut avoir une similarité légèrement
                        # inférieure à son seuil (ombre, vibration, petit reflet)
                        # et l'ancienne V4 la rejetait en INDETERMINE.
                        #
                        # Désormais :
                        # - estimer_etat() reste responsable de OUVERT / FERME ;
                        # - si l'image est loin des DEUX références, elle est
                        #   bloquée immédiatement (aucun comptage) ;
                        # - si en plus sa structure est inconnue pendant
                        #   plusieurs images, on confirme OCCLUSION.

                        structure_inconnue = (
                            similarite_max
                            < self.seuil_structure_occlusion
                        )

                        # Une image est "loin" si elle ne ressemble ni à OUVERT
                        # ni à FERME. Cela bloque immédiatement une main,
                        # un gros reflet ou une transition très anormale.
                        loin_des_refs = (
                            min(diff_ouverte, diff_fermee)
                            > max(45.0, self.seuil * 1.10)
                        )

                        occlusion_forte = (
                            self.protection_occlusion
                            and loin_des_refs
                            and structure_inconnue
                        )

                        if self.occlusion_active:
                            self.frames_occlusion += 1
                            self.etat = 'OCCLUSION'

                            # Sortie sûre : OUVERT doit être reconnu normalement
                            # pendant 3 images consécutives et ne plus être loin
                            # des références.
                            ouvert_valide = (
                                etat_brut == 'OUVERT'
                                and not loin_des_refs
                            )

                            if ouvert_valide:
                                self.retour_ouvert_candidat += 1
                            else:
                                self.retour_ouvert_candidat = 0

                            if self.retour_ouvert_candidat >= 3:
                                self.occlusion_active = False
                                self.occlusion_candidat = 0
                                self.retour_ouvert_candidat = 0
                                self.etat = 'OUVERT'
                                self.compteur.reprendre_apres_occlusion(
                                    now - self.debut_test
                                )

                        elif loin_des_refs:
                            # IMPORTANT :
                            # Dès la PREMIÈRE image loin de OUVERT et FERME,
                            # on la bloque. Elle ne peut donc pas devenir une
                            # fausse fermeture.
                            self.etat = 'INDETERMINE'

                            self.compteur.candidat = None
                            self.compteur.repetitions = 0
                            self.compteur.debut = None

                            if occlusion_forte:
                                self.occlusion_candidat += 1
                            else:
                                self.occlusion_candidat = 0

                            # On affiche OCCLUSION seulement après 3 images
                            # vraiment anormales consécutives.
                            if self.occlusion_candidat >= 3:
                                self.occlusion_active = True
                                self.occlusions_detectees += 1
                                self.frames_occlusion += 1
                                self.retour_ouvert_candidat = 0
                                self.etat = 'OCCLUSION'

                        else:
                            # Image suffisamment proche d'un état connu :
                            # on fait confiance à la détection OUVERT / FERME
                            # existante. Pas de filtre structurel supplémentaire.
                            self.occlusion_candidat = 0
                            self.etat = etat_brut

                            self.compteur.ajouter(
                                self.etat,
                                self.numero,
                                now - self.debut_test
                            )

        except Exception as exc:
            with self.lock:
                self.erreur = str(exc)
                self.actif = False

        finally:
            if cap is not None:
                cap.release()

    def demarrer_capture_calibration(self):
        """
        Démarre un enregistrement dédié à la calibration V6.

        Le buffer est indépendant de l'historique de 1200 images utilisé
        pour les références manuelles.
        """
        with self.lock:
            if self.actif:
                raise ValueError(
                    "Arrêtez le comptage avant de démarrer une calibration."
                )

            self.calibration_buffer.clear()
            self.calibration_buffer_sature = False
            self.calibration_started_monotonic = time.monotonic()
            self.calibration_started_at = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self.calibration_capture_active = True

            self.images_rejouees_calibration = 0
            self.fermetures_recuperees_calibration = 0
            self.cables_recuperes_calibration = 0

    def obtenir_images_calibration(self):
        """
        Retourne une copie des images de calibration sans les timestamps,
        au format attendu par calibrer_references_automatiques().
        """
        with self.lock:
            return [
                (numero, image.copy())
                for numero, image, _ in self.calibration_buffer
            ]

    def annuler_capture_calibration(self):
        """Arrête et efface la capture de calibration."""
        with self.lock:
            self.calibration_capture_active = False
            self.calibration_buffer.clear()
            self.calibration_buffer_sature = False
            self.calibration_started_monotonic = None
            self.calibration_started_at = None

    @staticmethod
    def _rejouer_calibration(
        frames,
        refs_preparees,
        marge,
        seuil,
        compteur,
        origine_s
    ):
        """
        Rejoue chronologiquement les ROI capturées pendant la calibration.

        IMPORTANT :
        - on réutilise exactement estimer_etat() ;
        - INDETERMINE ne crée jamais une fermeture ;
        - la détection de pause est volontairement désactivée pendant le
          rattrapage. Une commande longue ne doit pas être confondue avec une
          pause simplement parce que le câble met du temps à avancer.
        """
        dernier_numero = None
        dernier_etat = None

        for numero, zone, timestamp in frames:
            gris = cv2.cvtColor(
                zone,
                cv2.COLOR_RGB2GRAY
            )

            etat, _, _ = estimer_etat(
                gris,
                refs_preparees,
                marge,
                seuil
            )

            compteur.ajouter(
                etat,
                numero,
                max(0.0, timestamp - origine_s)
            )

            dernier_numero = numero
            dernier_etat = etat

        return dernier_numero, dernier_etat

    def demarrer(
        self,
        refs,
        marge,
        seuil,
        confirmation,
        test_meta=None,
        ignorer_initiales=2,
        seuil_pause_s=5.0,
        recuperer_calibration=False
    ):
        """
        Démarre le comptage.

        V6 :
        si recuperer_calibration=True, les images capturées depuis le début de
        la calibration sont d'abord rejouées dans un Compteur local. L'état du
        compteur obtenu devient ensuite l'état de départ du comptage temps réel.
        """

        # ------------------------------------------------------
        # 1) Vérification + première photographie du buffer
        # ------------------------------------------------------
        with self.lock:
            if self.erreur or not self.thread.is_alive():
                raise ValueError(
                    'Reconnectez la caméra avant de démarrer.'
                )

            if recuperer_calibration and self.calibration_buffer_sature:
                raise ValueError(
                    "Le buffer de calibration a atteint sa limite. "
                    "Relancez une calibration puis démarrez le comptage "
                    "plus rapidement afin de garantir qu'aucun mouvement "
                    "du début de commande n'est perdu."
                )

            frames_initiales = (
                [
                    (n, z.copy(), t)
                    for n, z, t in self.calibration_buffer
                ]
                if recuperer_calibration
                else []
            )

            origine_calibration = (
                self.calibration_started_monotonic
                if recuperer_calibration
                else None
            )

            date_calibration = (
                self.calibration_started_at
                if recuperer_calibration
                else None
            )

        # ------------------------------------------------------
        # 2) Préparation des références
        # ------------------------------------------------------
        refs_preparees = preparer_references(refs)

        refs_brutes = {
            etat: [
                cv2.cvtColor(
                    ref["image"],
                    cv2.COLOR_RGB2GRAY
                )
                for ref in refs[etat]
            ]
            for etat in ("ouvert", "ferme")
        }

        refs_structure = {
            etat: [
                extraire_signature_structure(
                    ref["image"]
                )
                for ref in refs[etat]
            ]
            for etat in ("ouvert", "ferme")
        }

        if any(
            (a == b).all()
            for a in refs_preparees['ouvert']
            for b in refs_preparees['ferme']
        ):
            raise ValueError(
                'Les références ouverte et fermée sont identiques.'
            )

        # Seuils structurels utilisés par la protection OCCLUSION temps réel.
        seuils_par_etat = {}

        for etat in ("ouvert", "ferme"):
            exemples = refs_structure[etat]
            similarites_etat = []

            for i in range(len(exemples)):
                for j in range(i + 1, len(exemples)):
                    similarites_etat.append(
                        similarite_structure(
                            exemples[i],
                            exemples[j]
                        )
                    )

            if similarites_etat:
                minimum_valide = min(similarites_etat)

                seuils_par_etat[etat] = max(
                    0.55,
                    min(
                        0.80,
                        minimum_valide - 0.08
                    )
                )
            else:
                seuils_par_etat[etat] = 0.60

        seuil_validation_ouvert = seuils_par_etat["ouvert"]
        seuil_validation_ferme = seuils_par_etat["ferme"]

        seuil_structure_occlusion = max(
            0.45,
            min(
                seuil_validation_ouvert,
                seuil_validation_ferme
            ) - 0.05
        )

        # ------------------------------------------------------
        # 3) Créer le compteur et rejouer la calibration
        # ------------------------------------------------------
        # Pendant le replay, la pause temporelle est désactivée :
        # la longueur du câble ne doit pas fausser F1/F2/F3.
        compteur_nouveau = Compteur(
            confirmation=confirmation,
            ignorer_initiales=ignorer_initiales,
            seuil_pause_s=1e9 if recuperer_calibration else seuil_pause_s
        )

        dernier_numero_rejoue = None
        dernier_etat_rejoue = None
        frames_rejouees = []

        if recuperer_calibration:
            if not frames_initiales or origine_calibration is None:
                raise ValueError(
                    "Aucune séquence de calibration disponible à récupérer."
                )

            dernier_numero_rejoue, dernier_etat_rejoue = (
                self._rejouer_calibration(
                    frames_initiales,
                    refs_preparees,
                    marge,
                    seuil,
                    compteur_nouveau,
                    origine_calibration
                )
            )
            frames_rejouees.extend(frames_initiales)

            # Deuxième passe de rattrapage :
            # pendant le premier replay, la caméra continue d'enregistrer.
            with self.lock:
                suite = [
                    (n, z.copy(), t)
                    for n, z, t in self.calibration_buffer
                    if (
                        dernier_numero_rejoue is None
                        or n > dernier_numero_rejoue
                    )
                ]

            if suite:
                dernier_numero_rejoue, dernier_etat_rejoue = (
                    self._rejouer_calibration(
                        suite,
                        refs_preparees,
                        marge,
                        seuil,
                        compteur_nouveau,
                        origine_calibration
                    )
                )
                frames_rejouees.extend(suite)

        # ------------------------------------------------------
        # 4) Bascule atomique vers le temps réel
        # ------------------------------------------------------
        with self.lock:
            # Dernière petite queue arrivée entre le second replay et ce lock.
            if recuperer_calibration:
                queue_finale = [
                    (n, z.copy(), t)
                    for n, z, t in self.calibration_buffer
                    if (
                        dernier_numero_rejoue is None
                        or n > dernier_numero_rejoue
                    )
                ]

                if queue_finale:
                    dernier_numero_rejoue, dernier_etat_rejoue = (
                        self._rejouer_calibration(
                            queue_finale,
                            refs_preparees,
                            marge,
                            seuil,
                            compteur_nouveau,
                            origine_calibration
                        )
                    )
                    frames_rejouees.extend(queue_finale)

                self.calibration_capture_active = False

            # Après le replay, on réactive le seuil de pause choisi pour
            # le temps réel.
            compteur_nouveau.seuil_pause_s = float(seuil_pause_s)

            # Marquer les événements déjà récupérés.
            for evenement in compteur_nouveau.evenements:
                evenement["Source"] = "CALIBRATION"

            self.refs = refs_preparees
            self.refs_brutes = refs_brutes
            self.refs_structure = refs_structure

            self.seuil_validation_ouvert = seuil_validation_ouvert
            self.seuil_validation_ferme = seuil_validation_ferme
            self.seuil_structure_occlusion = seuil_structure_occlusion

            self.occlusion_active = False
            self.occlusion_candidat = 0
            self.retour_ouvert_candidat = 0
            self.occlusions_detectees = 0
            self.frames_occlusion = 0

            self.compteur = compteur_nouveau
            self.etat = (
                dernier_etat_rejoue
                if recuperer_calibration and dernier_etat_rejoue
                else 'NON INITIALISE'
            )

            self.marge = marge
            self.seuil = seuil
            self.confirmation_test = confirmation

            maintenant = time.monotonic()

            self.debut_test = (
                origine_calibration
                if recuperer_calibration and origine_calibration is not None
                else maintenant
            )

            self.test_meta = dict(test_meta or {})

            self.test_started_at = (
                date_calibration
                if recuperer_calibration and date_calibration
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            self.test_ended_at = None
            self.test_started_monotonic = self.debut_test
            self.test_ended_monotonic = None

            self.of_compteur = self.code_barres
            self.scan_barcode_actif = False

            self.images_rejouees_calibration = len(frames_rejouees)
            self.fermetures_recuperees_calibration = len(
                self.compteur.evenements
            )
            self.cables_recuperes_calibration = self.compteur.cables

            # Le buffer n'est plus nécessaire après la bascule.
            self.calibration_buffer.clear()
            self.calibration_buffer_sature = False

            self.actif = True

    def arreter(self):
        with self.lock:
            if self.actif:
                self.test_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.test_ended_monotonic = time.monotonic()
            self.actif = False

    def nouveau_test(self):
        """
        Remet uniquement les résultats du test à zéro.
        Les références OUVERT/FERMÉ et la position caméra restent inchangées.
        """
        with self.lock:
            if self.actif:
                raise ValueError(
                    "Arrêtez le comptage avant de préparer un nouveau test."
                )

            confirmation = self.confirmation_test or 1
            self.compteur = Compteur(confirmation=confirmation)
            self.etat = 'NON INITIALISE'
            self.of_compteur = None
            self.test_meta = {}
            self.test_started_at = None
            self.test_ended_at = None
            self.test_started_monotonic = None
            self.test_ended_monotonic = None

            self.images_rejouees_calibration = 0
            self.fermetures_recuperees_calibration = 0
            self.cables_recuperes_calibration = 0

    def nouveau_of(self):
        with self.lock:

            if self.actif:
                raise ValueError(
                    "Arrêtez le comptage avant de scanner un nouvel OF."
                )

            self.code_barres = None
            self.code_barres_format = None
            self.of_compteur = None

            # Réactive la recherche automatique.
            self.scan_barcode_actif = True

    def fermer(self):
        self.arreter()
        self.stop_event.set()
        self.thread.join(timeout=2)


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

def afficher_camera():

    if not hasattr(st, 'fragment'):
        st.error(
            'Mettez Streamlit à jour : '
            'python -m pip install --upgrade streamlit'
        )
        return

    # ========================================================
    # V8.1 — WORK ORDER FACULTATIF AVANT LE COMPTEUR
    # ========================================================
    if "wo_valide" not in st.session_state:
        st.session_state.wo_valide = False

    if "commande_validee" not in st.session_state:
        st.session_state.commande_validee = False

    if "work_order" not in st.session_state:
        st.session_state.work_order = ""

    # --------------------------------------------------------
    # ETAPE 1 : scan FACULTATIF du Work Order — AUTOMATIQUE
    # --------------------------------------------------------
    if not st.session_state.wo_valide:
        st.subheader("1 — Scanner le Work Order (facultatif)")

        st.info(
            "Scannez directement le Work Order avec le scanner USB. "
            "Aucun champ à sélectionner et aucun bouton de validation : "
            "le scan est capté automatiquement puis l'application passe "
            "à l'étape Longueur + Quantité."
        )

        # Le callback s'exécute AVANT le rerun Streamlit.
        # On ne modifie jamais wo_scan_input après création du widget :
        # cela évite StreamlitWidgetAlreadyInstantiatedError.
        def traiter_scan_work_order():
            brut = st.session_state.get("wo_scan_input", "")
            wo = normaliser_work_order_scanner(brut)

            if wo is None:
                st.session_state["wo_scan_erreur"] = (
                    "Work Order invalide. Format attendu : 48225-1. "
                    "Rescannez le code."
                )
                return

            st.session_state.work_order = wo
            st.session_state.wo_valide = True
            st.session_state.pop("wo_scan_erreur", None)

        # Champ technique invisible :
        # le scanner USB se comporte comme un clavier, il lui faut donc un
        # élément ayant le focus. On garde ce champ hors écran et on lui donne
        # automatiquement le focus avec JavaScript.
        st.markdown(
            """
            <style>
            div[data-testid="stTextInput"]:has(
                input[aria-label="WO_SCANNER_HIDDEN"]
            ) {
                position: fixed !important;
                left: -10000px !important;
                top: -10000px !important;
                width: 1px !important;
                height: 1px !important;
                opacity: 0 !important;
                overflow: hidden !important;
                pointer-events: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.text_input(
            "WO_SCANNER_HIDDEN",
            key="wo_scan_input",
            on_change=traiter_scan_work_order,
            label_visibility="collapsed"
        )

        # Autofocus automatique sur le champ scanner invisible.
        # On réessaie plusieurs fois car Streamlit peut recréer le DOM.
        components.html(
            """
            <script>
            (function () {
                function focusScanner() {
                    try {
                        const doc = window.parent.document;
                        const el = doc.querySelector(
                            'input[aria-label="WO_SCANNER_HIDDEN"]'
                        );
                        if (el) {
                            el.focus();
                            el.setAttribute('autocomplete', 'off');
                            return true;
                        }
                    } catch (e) {
                        // Le retry ci-dessous tentera à nouveau.
                    }
                    return false;
                }

                let essais = 0;
                const timer = setInterval(() => {
                    essais += 1;
                    if (focusScanner() || essais >= 30) {
                        clearInterval(timer);
                    }
                }, 100);

                setTimeout(focusScanner, 50);
            })();
            </script>
            """,
            height=0,
            width=0
        )

        erreur_scan = st.session_state.get("wo_scan_erreur")
        if erreur_scan:
            st.error(erreur_scan)

        st.caption(
            "Le scanner doit envoyer Entrée/CR à la fin du code."
        )

        if st.button("Continuer sans Work Order"):
            st.session_state.work_order = ""
            st.session_state.wo_valide = True
            st.session_state.pop("wo_scan_erreur", None)
            st.rerun()

        return

    # --------------------------------------------------------
    # ETAPE 2 : longueur + quantité obligatoires
    # --------------------------------------------------------
    if not st.session_state.commande_validee:
        wo = st.session_state.work_order

        if wo:
            st.success(f"Work Order scanné : {wo}")
        else:
            st.info("Aucun Work Order scanné — mode sans WO.")

        st.subheader("2 — Informations de la commande")

        with st.form("form_parametres_commande"):
            longueur = st.number_input(
                "Longueur du câble (m)",
                min_value=0.01,
                value=1.10,
                step=0.10,
                format="%.2f"
            )

            quantite = st.number_input(
                "Quantité à produire",
                min_value=1,
                value=1,
                step=1
            )

            valider_commande = st.form_submit_button(
                "Valider la commande et ouvrir le compteur",
                type="primary"
            )

        if valider_commande:
            st.session_state.test_id = (
                wo if wo else "SANS_WO"
            )
            st.session_state.test_longueur = float(longueur)
            st.session_state.test_quantite_cible = int(quantite)
            st.session_state.test_commentaire = ""
            st.session_state.commande_validee = True
            st.rerun()

        return

    work_order = st.session_state.work_order

    wo_affiche = work_order if work_order else "Non scanné"

    st.success(
        f"Commande active — WO : {wo_affiche} | "
        f"Longueur : {st.session_state.test_longueur:.2f} m | "
        f"Quantité : {st.session_state.test_quantite_cible}"
    )

    cam = st.session_state.get('camera_direct')

    if st.button(
        "Nouvelle commande",
        disabled=bool(cam is not None and cam.actif)
    ):
        if cam is not None:
            cam.fermer()
            st.session_state.pop("camera_direct", None)

        for cle in (
            "cam_refs",
            "cam_gel",
            "auto_calib_start_num",
            "auto_calib_en_cours",
            "auto_calib_rapport",
            "auto_calib_erreur",
            "auto_global_config",
            "auto_global_processing",
            "wo_scan_input",
            "wo_scan_erreur",
        ):
            st.session_state.pop(cle, None)

        st.session_state.work_order = ""
        st.session_state.wo_valide = False
        st.session_state.commande_validee = False
        st.rerun()

    st.info(
        'Caméra du PC qui exécute Streamlit. '
        'Fermez les autres applications utilisant la caméra.'
    )

    cam = st.session_state.get('camera_direct')
    connectee = cam is not None

    index = int(
        st.number_input(
            'Index caméra (0, puis 1 ou 2 si nécessaire)',
            0,
            10,
            0,
            disabled=connectee
        )
    )

    rotation = st.selectbox(
        'Orientation de la caméra',
        ['Aucune', '90° droite', '90° gauche', '180°'],
        index=2,
        disabled=connectee,
        help=(
            "Pour la fixation actuelle de la machine, 90° gauche est "
            "sélectionné par défaut. Déconnectez la caméra pour changer "
            "l'orientation si nécessaire."
        )
    )

    # V8 : le Work Order vient du scanner USB obligatoire.
    activer_barcode = False

    verrouiller_exposition = st.checkbox(
        "Verrouiller l'exposition caméra après connexion",
        value=True,
        disabled=connectee,
        help=(
            "La caméra se stabilise d'abord sous la lumière normale, "
            "puis l'application tente de figer l'exposition actuelle. "
            "Cela limite les changements automatiques quand l'éclairage varie."
        )
    )

    gx, dx = st.slider(
        'Zone horizontale (%)',
        0,
        100,
        (31, 43),
        disabled=connectee
    )

    hy, by = st.slider(
        'Zone verticale (%)',
        0,
        100,
        (36, 48),
        disabled=connectee
    )

    st.caption(
        'Fixation validée : orientation 90° gauche — '
        'ROI horizontale 31–43 % — ROI verticale 36–48 %. '
        'Pour modifier la zone, l’orientation ou changer de caméra, '
        'déconnectez puis reconnectez. Les références seront effacées.'
    )

    if not connectee:
        st.info(
            "Fixation actuelle : orientation recommandée « 90° gauche ». "
            "Après connexion, l'image doit passer de 1280×720 à 720×1280."
        )

    if st.button(
        'Connecter la caméra',
        disabled=connectee
    ):
        if gx >= dx or hy >= by:
            st.error(
                'Choisissez une zone non vide.'
            )
        else:
            st.session_state.camera_direct = Camera(
                index,
                (gx, dx, hy, by),
                rotation=rotation,
                activer_barcode=activer_barcode,
                verrouiller_exposition=verrouiller_exposition
            )

            # V8.1 : le Work Order est facultatif.
            if work_order:
                st.session_state.camera_direct.code_barres = work_order
                st.session_state.camera_direct.code_barres_format = "SCANNER_USB"
            else:
                st.session_state.camera_direct.code_barres = None
                st.session_state.camera_direct.code_barres_format = None

            st.session_state.camera_direct.scan_barcode_actif = False

            st.session_state.cam_refs = {
                'ouvert': [],
                'ferme': []
            }

            st.session_state.pop(
                'cam_gel',
                None
            )

            # Nouvelle connexion = nouvelle session de calibration automatique.
            st.session_state.pop('auto_calib_start_num', None)
            st.session_state.pop('auto_calib_en_cours', None)
            st.session_state.pop('auto_calib_rapport', None)
            st.session_state.pop('auto_calib_erreur', None)
            st.session_state.pop('auto_global_config', None)
            st.session_state.pop('auto_global_processing', None)

            st.rerun()

    if cam is None:
        return

    if st.button(
        'Déconnecter la caméra'
    ):
        cam.fermer()
        del st.session_state.camera_direct
        st.rerun()

    # ========================================================
    # COMMANDE DE PRODUCTION VERROUILLÉE PAR LE SCANNER
    # ========================================================

    st.subheader("Commande de production")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Work Order",
        st.session_state.work_order or "Non scanné"
    )

    c2.metric(
        "Longueur câble",
        f"{st.session_state.test_longueur:.2f} m"
    )

    c3.metric(
        "Quantité prévue",
        int(st.session_state.test_quantite_cible)
    )

    st.text_input(
        "Commentaire (optionnel)",
        key="test_commentaire",
        placeholder="Ex. production normale, remarque opérateur..."
    )

    st.divider()

    # ========================================================
    # PANNEAU TEMPS REEL
    # ========================================================

    @st.fragment(run_every=0.2)
    def panneau():

        cam.heartbeat = time.monotonic()

        with cam.lock:
            actif = cam.actif
            dernier = cam.derniere
            erreur = cam.erreur

            evenements = list(
                cam.compteur.evenements
            )
            cables_comptes = cam.compteur.cables
            phase_machine = cam.compteur.phase
            pauses_detectees = cam.compteur.pauses_detectees
            initiales_ignorees = (
                cam.compteur.fermetures_ignorees_initialisation
            )
            reprise_ignorees = (
                cam.compteur.fermetures_ignorees_reprise
            )

            images_rejouees_calibration = (
                cam.images_rejouees_calibration
            )
            fermetures_recuperees_calibration = (
                cam.fermetures_recuperees_calibration
            )
            cables_recuperes_calibration = (
                cam.cables_recuperes_calibration
            )

            occlusion_active = cam.occlusion_active
            occlusions_detectees = cam.occlusions_detectees
            frames_occlusion = cam.frames_occlusion
            fraction_changement = cam.fraction_changement
            diff_ouverte = cam.diff_ouverte
            diff_fermee = cam.diff_fermee
            similarite_structure_max = cam.similarite_structure_max
            seuil_structure_occlusion = cam.seuil_structure_occlusion
            similarite_ouverte = cam.similarite_ouverte
            similarite_fermee = cam.similarite_fermee
            seuil_validation_ouvert = cam.seuil_validation_ouvert
            seuil_validation_ferme = cam.seuil_validation_ferme

            incomplet = (
                cam.compteur.debut
                is not None
            )

            initiale = (
                cam.compteur.initiale
            )

            etat = cam.etat
            numero = cam.numero

            code_barres = cam.code_barres
            code_barres_format = cam.code_barres_format
            scan_barcode_actif = cam.scan_barcode_actif
            of_compteur = cam.of_compteur

        if erreur:
            st.error(erreur)

        if dernier is None:
            st.info(
                'Ouverture de la caméra…'
            )
            return

        image, zone, (x1, y1, x2, y2) = dernier

        hauteur_affichee, largeur_affichee = image.shape[:2]

        st.caption(
            f"Orientation appliquée : {cam.rotation} — "
            f"image utilisée : {largeur_affichee} × {hauteur_affichee}"
        )

        if cam.verrouiller_exposition:
            if cam.exposition_verrouillee:
                st.caption(
                    "Exposition caméra : verrouillage manuel demandé ✅ "
                    f"(avant : {cam.exposition_capturee}, "
                    f"actuelle : {cam.exposition_reelle}, "
                    f"auto : {cam.auto_exposure_reelle})"
                )
            else:
                st.warning(
                    "Le pilote de cette caméra n'a pas confirmé le "
                    "verrouillage de l'exposition. Le comptage fonctionne, "
                    "mais l'exposition peut encore varier automatiquement."
                )
        else:
            st.caption("Exposition caméra : automatique.")

        apercu = image.copy()

        # Rectangle vert = ROI du compteur.
        cv2.rectangle(
            apercu,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2
        )

        # Affiche aussi l'OF sur l'aperçu.
        if code_barres:
            cv2.putText(
                apercu,
                f"OF : {code_barres}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        a, b = st.columns(2)

        a.image(
            apercu,
            channels='BGR',
            width=440
        )

        b.image(
            zone,
            width=260,
            caption='Zone analysée pour le comptage'
        )

        cadence = (
            numero
            /
            max(
                0.001,
                time.monotonic() - cam.depart
            )
        )

        fps_pilote = cam.fps_pilote

        st.caption(
            f'{numero} images lues. '
            f'Caméra configurée à {cam.fps_demande:.0f} FPS'
            + (
                f' — pilote annonce {fps_pilote:.1f} FPS.'
                if fps_pilote is not None
                else '.'
            )
            + f' Boucle OpenCV observée : {cadence:.1f} images/s. '
            'Cette dernière valeur inclut lecture + traitement et '
            'ne signifie pas que la caméra est limitée à ce FPS.'
        )

        # ====================================================
        # WORK ORDER — SCANNER USB OBLIGATOIRE
        # ====================================================

        st.subheader("Work Order")

        if st.session_state.work_order:
            st.success(
                f"WO : {st.session_state.work_order}"
            )
            st.caption(
                "Source : scanner USB. Le Work Order ne peut pas être changé "
                "pendant le comptage."
            )
        else:
            st.info(
                "Aucun Work Order scanné pour cette commande."
            )

        refs = st.session_state.cam_refs

        # ====================================================
        # PARAMETRAGE AVANT COMPTAGE
        # ====================================================

        if not actif:

            st.subheader("Compteur global automatique — V7")

            st.caption(
                "Un seul démarrage : l'application enregistre 800 images, "
                "crée automatiquement 2 références OUVERT + 2 références "
                "FERMÉ, récupère les câbles déjà coupés pendant cette phase, "
                "puis continue le même compteur en temps réel."
            )

            # ----------------------------------------------------
            # PARAMÈTRES DU COMPTAGE
            # ----------------------------------------------------
            auto_en_cours = bool(
                st.session_state.get("auto_calib_en_cours", False)
            )

            marge = st.number_input(
                'Marge minimale',
                0.0,
                20.0,
                0.5,
                0.1,
                disabled=auto_en_cours
            )

            seuil = st.number_input(
                'Différence maximale',
                1.0,
                100.0,
                40.0,
                1.0,
                disabled=auto_en_cours
            )

            confirmation = st.number_input(
                'Images de confirmation',
                1,
                10,
                1,
                disabled=auto_en_cours,
                help=(
                    "Valeur recommandée : 1, afin de ne pas rater les "
                    "fermetures très rapides."
                )
            )

            nouvelle_commande = st.checkbox(
                "Début d'une nouvelle commande : ignorer les 2 mouvements d'initialisation",
                value=True,
                disabled=auto_en_cours,
                help=(
                    "Les 2 premiers mouvements de mise en route sont ignorés. "
                    "Ensuite la séquence repart sur F1 / F2 / F3."
                )
            )

            seuil_pause_s = st.number_input(
                "Pause / resynchronisation après (secondes)",
                min_value=2.0,
                max_value=120.0,
                value=5.0,
                step=1.0,
                disabled=auto_en_cours,
                help=(
                    "Ce seuil reste utilisé uniquement après le passage au "
                    "comptage temps réel. Pendant le replay des 800 images, "
                    "la pause temporelle est désactivée."
                )
            )

            # ----------------------------------------------------
            # MODE AUTOMATIQUE COMPLET
            # ----------------------------------------------------
            if auto_en_cours:
                images_calibration = cam.obtenir_images_calibration()
                nb_images_auto = len(images_calibration)

                progression = min(
                    1.0,
                    nb_images_auto / float(AUTO_CALIBRATION_CIBLE_IMAGES)
                )

                st.progress(progression)
                st.metric(
                    "Images enregistrées automatiquement",
                    f"{min(nb_images_auto, AUTO_CALIBRATION_CIBLE_IMAGES)} / "
                    f"{AUTO_CALIBRATION_CIBLE_IMAGES}"
                )

                if nb_images_auto < AUTO_CALIBRATION_CIBLE_IMAGES:
                    st.info(
                        "Compteur en préparation automatique : continuez la "
                        "production normalement. Ne cliquez sur rien. Les "
                        "câbles coupés maintenant seront récupérés ensuite."
                    )

                    if st.button("❌ Annuler le démarrage automatique"):
                        cam.annuler_capture_calibration()
                        st.session_state["auto_calib_en_cours"] = False
                        st.session_state.pop("auto_global_config", None)
                        st.session_state.pop("auto_global_processing", None)
                        st.rerun(scope="fragment")

                elif not st.session_state.get("auto_global_processing", False):
                    # On verrouille immédiatement cette étape pour empêcher
                    # deux traitements du même seuil de 800 images.
                    st.session_state["auto_global_processing"] = True

                    config_auto = st.session_state.get(
                        "auto_global_config",
                        {}
                    )

                    try:
                        with st.spinner(
                            "800 images atteintes : création automatique des "
                            "2 OUVERT + 2 FERMÉ et récupération des câbles..."
                        ):
                            # Les références sont choisies exactement dans les
                            # 800 premières images demandées par l'opérateur.
                            images_pour_refs = images_calibration[
                                :AUTO_CALIBRATION_CIBLE_IMAGES
                            ]

                            refs_auto, rapport_auto = (
                                calibrer_references_automatiques(
                                    images_pour_refs,
                                    nb_refs=AUTO_CALIBRATION_NB_REFS,
                                    nb_images_ouvert_initial=15
                                )
                            )

                            # On ne remplace les références qu'après réussite.
                            refs["ouvert"].clear()
                            refs["ferme"].clear()
                            refs["ouvert"].extend(refs_auto["ouvert"])
                            refs["ferme"].extend(refs_auto["ferme"])

                            rapport_auto["images_cible"] = (
                                AUTO_CALIBRATION_CIBLE_IMAGES
                            )
                            rapport_auto["nb_refs_par_etat"] = (
                                AUTO_CALIBRATION_NB_REFS
                            )

                            # Démarrage IMMÉDIAT du compteur global.
                            # demarrer() rejoue tout le buffer capturé depuis le
                            # clic initial, y compris les images arrivées pendant
                            # le calcul de calibration, puis bascule en temps réel.
                            cam.demarrer(
                                refs,
                                float(config_auto.get("marge", 0.5)),
                                float(config_auto.get("seuil", 40.0)),
                                int(config_auto.get("confirmation", 1)),
                                test_meta=dict(
                                    config_auto.get("test_meta", {})
                                ),
                                ignorer_initiales=(
                                    2
                                    if config_auto.get(
                                        "nouvelle_commande",
                                        True
                                    )
                                    else 0
                                ),
                                seuil_pause_s=float(
                                    config_auto.get("seuil_pause_s", 5.0)
                                ),
                                recuperer_calibration=True
                            )

                            st.session_state["auto_calib_rapport"] = (
                                rapport_auto
                            )
                            st.session_state["auto_calib_en_cours"] = False
                            st.session_state["auto_global_processing"] = False
                            st.session_state.pop("auto_calib_erreur", None)
                            st.session_state.pop("auto_global_config", None)

                        st.rerun(scope="fragment")

                    except Exception as exc:
                        cam.annuler_capture_calibration()

                        st.session_state["auto_calib_erreur"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        st.session_state["auto_calib_en_cours"] = False
                        st.session_state["auto_global_processing"] = False
                        st.session_state.pop("auto_global_config", None)
                        st.rerun(scope="fragment")

            else:
                erreur_auto = st.session_state.get("auto_calib_erreur")

                if erreur_auto:
                    st.error(
                        "Démarrage automatique impossible : "
                        f"{erreur_auto}"
                    )

                rapport_auto = st.session_state.get("auto_calib_rapport")

                if rapport_auto and len(refs["ouvert"]) >= 2 and len(refs["ferme"]) >= 2:
                    st.success(
                        "Dernière calibration automatique : "
                        "2 OUVERT + 2 FERMÉ validés."
                    )

                    st.caption(
                        "Séparation OUVERT/FERMÉ : "
                        f"{rapport_auto['separation_min_ouvert_ferme']:.2f} — "
                        "pics mécaniques candidats : "
                        f"{rapport_auto.get('pics_mecaniques_detectes', '?')} — "
                        "OUVERT : "
                        f"{rapport_auto['numeros_ouvert']} — "
                        "FERMÉ : "
                        f"{rapport_auto['numeros_ferme']}."
                    )

                    cols_o = st.columns(2)
                    for i, ref in enumerate(refs["ouvert"][:2]):
                        cols_o[i].image(
                            ref["image"],
                            width=170,
                            caption=f"OUVERT auto #{ref['numero']}"
                        )

                    cols_f = st.columns(2)
                    for i, ref in enumerate(refs["ferme"][:2]):
                        cols_f[i].image(
                            ref["image"],
                            width=170,
                            caption=f"FERMÉ auto #{ref['numero']}"
                        )

                st.info(
                    "Avant de cliquer, laissez simplement la lame en position "
                    "OUVERT. Après le clic, vous continuez la production : "
                    "l'application fait le reste automatiquement."
                )

                if st.button(
                    "▶️ Démarrer le compteur global automatique",
                    disabled=bool(erreur),
                    type="primary"
                ):
                    try:
                        # Nouvelle commande = nouvelles références.
                        refs["ouvert"].clear()
                        refs["ferme"].clear()

                        test_meta = {
                            'Test_ID': st.session_state.get(
                                'test_id',
                                'TEST_001'
                            ),
                            'Quantite_cible_cables': int(
                                st.session_state.get(
                                    'test_quantite_cible',
                                    1
                                )
                            ),
                            'Longueur_cable_m': float(
                                st.session_state.get(
                                    'test_longueur',
                                    0.0
                                )
                            ),
                            'Commentaire': st.session_state.get(
                                'test_commentaire',
                                ''
                            ),
                        }

                        # On fige tous les paramètres au moment du seul clic.
                        st.session_state["auto_global_config"] = {
                            "marge": float(marge),
                            "seuil": float(seuil),
                            "confirmation": int(confirmation),
                            "nouvelle_commande": bool(nouvelle_commande),
                            "seuil_pause_s": float(seuil_pause_s),
                            "test_meta": test_meta,
                        }

                        cam.demarrer_capture_calibration()

                        st.session_state["auto_calib_en_cours"] = True
                        st.session_state["auto_global_processing"] = False
                        st.session_state.pop("auto_calib_rapport", None)
                        st.session_state.pop("auto_calib_erreur", None)

                        st.rerun(scope="fragment")

                    except ValueError as exc:
                        st.error(str(exc))

                # ------------------------------------------------
                # MODE MANUEL DE SECOURS — conservé mais non nécessaire
                # pour le fonctionnement normal V7.
                # ------------------------------------------------
                with st.expander("Mode manuel de secours"):
                    st.caption(
                        "Utilisez cette partie uniquement si vous voulez "
                        "forcer manuellement les références."
                    )

                    with cam.lock:
                        nb_images_refs = len(cam.historique)

                    st.caption(
                        f"Images disponibles : {nb_images_refs} / 1200."
                    )

                    if st.button(
                        'Figer les images récentes — secours',
                        disabled=bool(erreur),
                        key='figer_secours_v7'
                    ):
                        with cam.lock:
                            st.session_state.cam_gel = list(cam.historique)
                        st.session_state.pop('cam_selection_secours_v7', None)

                    gel = st.session_state.get('cam_gel', [])

                    if gel:
                        selection = st.select_slider(
                            'Image à classer — secours',
                            options=list(range(len(gel))),
                            key='cam_selection_secours_v7'
                        )

                        n, z = gel[selection]
                        st.image(
                            z,
                            width=280,
                            caption=f'Image {n} — sélection figée'
                        )

                        c_ouv, c_fer = st.columns(2)

                        with c_ouv:
                            if st.button(
                                'Ajouter OUVERT — secours',
                                key='ajout_ouvert_secours_v7'
                            ):
                                refs['ouvert'].append({
                                    'numero': n,
                                    'image': z.copy()
                                })

                        with c_fer:
                            if st.button(
                                'Ajouter FERMÉ — secours',
                                key='ajout_ferme_secours_v7'
                            ):
                                refs['ferme'].append({
                                    'numero': n,
                                    'image': z.copy()
                                })

                    st.caption(
                        f"OUVERT : {len(refs['ouvert'])} — "
                        f"FERMÉ : {len(refs['ferme'])}."
                    )

                    if st.button(
                        "Démarrer avec les références manuelles",
                        disabled=(
                            bool(erreur)
                            or len(refs['ouvert']) < 1
                            or len(refs['ferme']) < 1
                        ),
                        key='demarrer_manuel_v7'
                    ):
                        try:
                            test_meta = {
                                'Test_ID': st.session_state.get(
                                    'test_id',
                                    'TEST_001'
                                ),
                                'Quantite_cible_cables': int(
                                    st.session_state.get(
                                        'test_quantite_cible',
                                        1
                                    )
                                ),
                                'Longueur_cable_m': float(
                                    st.session_state.get(
                                        'test_longueur',
                                        0.0
                                    )
                                ),
                                'Commentaire': st.session_state.get(
                                    'test_commentaire',
                                    ''
                                ),
                            }

                            cam.demarrer(
                                refs,
                                marge,
                                seuil,
                                confirmation,
                                test_meta=test_meta,
                                ignorer_initiales=(
                                    2 if nouvelle_commande else 0
                                ),
                                seuil_pause_s=seuil_pause_s,
                                recuperer_calibration=False
                            )
                            st.rerun(scope='fragment')

                        except ValueError as exc:
                            st.error(str(exc))

        else:

            if st.button(
                'Arrêter le comptage'
            ):
                cam.arreter()

                st.rerun(
                    scope='fragment'
                )

        # ====================================================
        # RESULTATS
        # ====================================================

        if of_compteur:
            st.info(
                f'OF du comptage : {of_compteur}'
            )

        st.write(
            f"Comptage : "
            f"{'EN COURS' if actif else 'ARRÊTÉ'} "
            f"— état estimé : {etat}"
        )

        if occlusion_active or etat == 'OCCLUSION':
            st.warning(
                "⚠️ Occlusion confirmée dans la ROI : "
                "comptage F1/F2/F3 GELÉ. "
                "La reprise se fera après 3 images OUVERT stables."
            )
        elif actif and etat == 'INDETERMINE':
            st.info(
                "Image hors des références OUVERT / FERMÉ : "
                "elle est ignorée pour protéger le comptage."
            )

        if actif and diff_ouverte is not None and diff_fermee is not None:
            st.caption(
                f"Protection ROI — "
                f"diff OUVERT : {diff_ouverte:.1f} | "
                f"diff FERMÉ : {diff_fermee:.1f} | "
                f"similarité structure max : "
                f"{(similarite_structure_max if similarite_structure_max is not None else 0.0):.2f} | "
                f"occlusions : {occlusions_detectees}."
            )

        st.metric(
            'Fermetures complètes',
            len(evenements)
        )

        st.metric(
            'Câbles comptés — coupe F2',
            cables_comptes
        )

        if images_rejouees_calibration > 0:
            st.success(
                "Rattrapage V6 : "
                f"{images_rejouees_calibration} images rejouées — "
                f"{fermetures_recuperees_calibration} fermeture(s) "
                f"récupérée(s) — "
                f"{cables_recuperes_calibration} câble(s) déjà "
                "récupéré(s) depuis la calibration."
            )

        noms_phase = {
            0: "F1 attendu — traçage",
            1: "F2 attendu — coupe",
            2: "F3 attendu — préparation suivant",
        }

        st.caption(
            f"Synchronisation machine : {noms_phase.get(phase_machine, 'inconnue')} — "
            f"pauses détectées : {pauses_detectees} — "
            f"initialisation commande ignorée : {initiales_ignorees} — "
            f"mouvements de reprise ignorés : {reprise_ignorees}."
        )

        if initiale:
            st.warning(
                'Début observé fermé : la première fermeture '
                'sans ouverture préalable est exclue.'
            )

        if incomplet and not actif:
            st.warning(
                'Une fermeture était encore en cours à l’arrêt.'
            )

        # ====================================================
        # BILAN DU TEST + 2 EXPORTS CLAIRS
        # ====================================================

        with cam.lock:
            test_meta = dict(cam.test_meta)
            test_started_at = cam.test_started_at
            test_ended_at = cam.test_ended_at
            test_started_monotonic = cam.test_started_monotonic
            test_ended_monotonic = cam.test_ended_monotonic
            marge_test = getattr(cam, 'marge', '')
            seuil_test = getattr(cam, 'seuil', '')
            confirmation_test = getattr(cam, 'confirmation_test', '')

        if not actif and test_started_at:
            st.subheader("Bilan du test")

            test_id_courant = test_meta.get('Test_ID', 'TEST')
            quantite_prevue = int(
                test_meta.get('Quantite_cible_cables', 1)
            )

            # La quantité réelle est la vérité terrain saisie par l'opérateur.
            cle_reelle = f"test_quantite_reelle_{test_id_courant}"

            if cle_reelle not in st.session_state:
                st.session_state[cle_reelle] = quantite_prevue

            quantite_reelle = int(
                st.number_input(
                    "Quantité réellement produite pendant ce test",
                    min_value=0,
                    step=1,
                    key=cle_reelle,
                    help=(
                        "C'est cette quantité qui sert au calcul de l'erreur. "
                        "Si le test a été arrêté avant la quantité prévue, "
                        "indiquez seulement la quantité réellement produite."
                    )
                )
            )

            # ------------------------------------------------
            # CALCULS PRINCIPAUX
            # ------------------------------------------------
            fermetures_detectees = len(evenements)
            cables_detectes = cables_comptes

            # IMPORTANT :
            # on ne calcule plus "3 fermetures = 1 câble".
            # Le nombre de câbles vient uniquement des événements F2_COUPE.
            ecart_cables = cables_detectes - quantite_reelle
            ecart_cables_abs = abs(ecart_cables)

            if quantite_reelle > 0:
                erreur_pct = (
                    ecart_cables_abs / quantite_reelle
                ) * 100.0
            else:
                erreur_pct = (
                    0.0 if cables_detectes == 0 else 100.0
                )

            objectif_atteint = (
                "OUI"
                if quantite_reelle >= quantite_prevue
                else "NON"
            )

            # Durée précise du test
            if (
                test_started_monotonic is not None
                and test_ended_monotonic is not None
            ):
                duree_test_s = max(
                    0.0,
                    test_ended_monotonic - test_started_monotonic
                )
            else:
                duree_test_s = 0.0

            # ------------------------------------------------
            # DUREES DES FERMETURES
            # ------------------------------------------------
            durees_fermeture = []

            for evenement in evenements:
                debut = evenement.get('Secondes_debut')
                fin = evenement.get('Secondes_reouverture')

                if debut is not None and fin is not None:
                    durees_fermeture.append(fin - debut)

            if durees_fermeture:
                duree_fermeture_moyenne = (
                    sum(durees_fermeture)
                    / len(durees_fermeture)
                )
                duree_fermeture_min = min(durees_fermeture)
                duree_fermeture_max = max(durees_fermeture)
            else:
                duree_fermeture_moyenne = None
                duree_fermeture_min = None
                duree_fermeture_max = None

            # ------------------------------------------------
            # AFFICHAGE SIMPLE DANS STREAMLIT
            # ------------------------------------------------
            a1, a2, a3 = st.columns(3)

            a1.metric(
                "Quantité réelle",
                quantite_reelle
            )

            a2.metric(
                "Câbles détectés",
                cables_detectes,
                delta=ecart_cables
            )

            a3.metric(
                "Erreur",
                f"{erreur_pct:.2f} %"
            )

            st.write(
                f"**Prévu :** {quantite_prevue} câbles  |  "
                f"**Réel :** {quantite_reelle}  |  "
                f"**Détecté :** {cables_detectes}"
            )

            st.write(
                f"**Fermetures complètes détectées :** {fermetures_detectees}  |  "
                f"**Coupes F2 comptées :** {cables_detectes}  |  "
                f"**Pauses/resynchronisations :** {pauses_detectees}"
            )

            st.write(
                f"**Initialisation commande ignorée :** {initiales_ignorees}  |  "
                f"**Mouvements de reprise ignorés :** {reprise_ignorees}  |  "
                f"**Occlusions détectées :** {occlusions_detectees}  |  "
                f"**Durée du test :** {duree_test_s:.1f} s"
            )

            # ------------------------------------------------
            # 1) BILAN CSV : UNE SEULE LIGNE
            # ------------------------------------------------
            date_export = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            bilan = {
                'Test_ID': test_id_courant,
                'Date_debut_test': test_started_at or '',
                'Date_fin_test': test_ended_at or '',
                'Date_export': date_export,
                'OF': of_compteur or code_barres or '',
                'Longueur_cable_m': test_meta.get(
                    'Longueur_cable_m',
                    ''
                ),
                'Quantite_prevue_cables': quantite_prevue,
                'Quantite_reelle_cables': quantite_reelle,
                'Cables_detectes': cables_detectes,
                'Ecart_cables': ecart_cables,
                'Ecart_cables_absolu': ecart_cables_abs,
                'Erreur_pct': round(erreur_pct, 4),
                'Fermetures_detectees': fermetures_detectees,
                'Coupes_F2_comptees': cables_detectes,
                'Pauses_resynchronisations': pauses_detectees,
                'Fermetures_initialisation_ignorees': initiales_ignorees,
                'Mouvements_reprise_ignores': reprise_ignorees,
                'Images_rejouees_calibration': images_rejouees_calibration,
                'Fermetures_recuperees_calibration': (
                    fermetures_recuperees_calibration
                ),
                'Cables_recuperes_calibration': (
                    cables_recuperes_calibration
                ),
                'Occlusions_detectees': occlusions_detectees,
                'Images_pendant_occlusion': frames_occlusion,
                'Phase_machine_fin': phase_machine,
                'Objectif_prevu_atteint': objectif_atteint,
                'Duree_test_s': round(duree_test_s, 3),
                'Duree_fermeture_moyenne_s': (
                    round(duree_fermeture_moyenne, 4)
                    if duree_fermeture_moyenne is not None
                    else ''
                ),
                'Duree_fermeture_min_s': (
                    round(duree_fermeture_min, 4)
                    if duree_fermeture_min is not None
                    else ''
                ),
                'Duree_fermeture_max_s': (
                    round(duree_fermeture_max, 4)
                    if duree_fermeture_max is not None
                    else ''
                ),
                'Orientation_camera': cam.rotation,
                'ROI_gauche_pct': cam.limites[0],
                'ROI_droite_pct': cam.limites[1],
                'ROI_haut_pct': cam.limites[2],
                'ROI_bas_pct': cam.limites[3],
                'Marge_minimale': marge_test,
                'Difference_maximale': seuil_test,
                'Images_confirmation': confirmation_test,
                'Commentaire': test_meta.get('Commentaire', ''),
            }

            st.subheader("Résumé du bilan")
            st.dataframe(
                [bilan],
                use_container_width=True,
                hide_index=True
            )

            fichier_bilan = io.StringIO()
            champs_bilan = list(bilan.keys())

            writer_bilan = csv.DictWriter(
                fichier_bilan,
                fieldnames=champs_bilan
            )
            writer_bilan.writeheader()
            writer_bilan.writerow(bilan)

            # ------------------------------------------------
            # 2) EVENEMENTS CSV : UNE LIGNE PAR FERMETURE
            # ------------------------------------------------
            champs_evenements = [
                'Test_ID',
                'OF',
                'Fermeture',
                'Source',
                'Etape_machine',
                'Cable_compte',
                'Pause_avant',
                'Image_debut',
                'Secondes_debut',
                'Image_reouverture',
                'Secondes_reouverture',
                'Duree_fermeture_s',
                'Image_confirmation',
            ]

            lignes_evenements = []

            for evenement in evenements:
                debut = evenement.get('Secondes_debut')
                fin = evenement.get('Secondes_reouverture')

                duree = (
                    round(fin - debut, 4)
                    if debut is not None and fin is not None
                    else ''
                )

                lignes_evenements.append({
                    'Test_ID': test_id_courant,
                    'OF': of_compteur or code_barres or '',
                    'Fermeture': evenement.get(
                        'Fermeture',
                        ''
                    ),
                    'Source': evenement.get(
                        'Source',
                        'TEMPS_REEL'
                    ),
                    'Etape_machine': evenement.get(
                        'Etape_machine',
                        ''
                    ),
                    'Cable_compte': evenement.get(
                        'Cable_compte',
                        ''
                    ),
                    'Pause_avant': evenement.get(
                        'Pause_avant',
                        ''
                    ),
                    'Image_debut': evenement.get(
                        'Image_debut',
                        ''
                    ),
                    'Secondes_debut': debut if debut is not None else '',
                    'Image_reouverture': evenement.get(
                        'Image_reouverture',
                        ''
                    ),
                    'Secondes_reouverture': (
                        fin if fin is not None else ''
                    ),
                    'Duree_fermeture_s': duree,
                    'Image_confirmation': evenement.get(
                        'Image_confirmation',
                        ''
                    ),
                })

            st.subheader("Événements détectés")

            if lignes_evenements:
                st.dataframe(
                    lignes_evenements,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "Aucune fermeture détectée pendant ce test."
                )

            fichier_evenements = io.StringIO()

            writer_evenements = csv.DictWriter(
                fichier_evenements,
                fieldnames=champs_evenements
            )
            writer_evenements.writeheader()
            writer_evenements.writerows(lignes_evenements)

            nom_test_sain = re.sub(
                r'[^A-Za-z0-9_-]+',
                '_',
                str(test_id_courant)
            )

            c_download_1, c_download_2 = st.columns(2)

            with c_download_1:
                st.download_button(
                    'Télécharger BILAN CSV',
                    fichier_bilan.getvalue().encode('utf-8-sig'),
                    f'{nom_test_sain}_BILAN.csv',
                    'text/csv',
                    key=f'download_bilan_{nom_test_sain}'
                )

            with c_download_2:
                st.download_button(
                    'Télécharger ÉVÉNEMENTS CSV',
                    fichier_evenements.getvalue().encode('utf-8-sig'),
                    f'{nom_test_sain}_EVENEMENTS.csv',
                    'text/csv',
                    key=f'download_evenements_{nom_test_sain}'
                )

    panneau()


# ============================================================
# INTERFACE PRODUCTION V1 — OPERATEUR MACHINE
# ============================================================

MACHINES_CONFIG = {
    "Machine 1": {
        "camera_index": 0,
        "rotation": "90° gauche",
        "roi": (31, 43, 36, 48),
        "verrouiller_exposition": True,
    },
    "Machine 2": {
        "camera_index": 0,
        "rotation": "90° droite",
        "roi": (29, 41, 45, 56),
        "verrouiller_exposition": True,
    },
}

PROD_AUTO_IMAGES = 800
PROD_NB_REFS = 2
PROD_MARGE = 0.5
PROD_SEUIL = 40.0
PROD_CONFIRMATION = 1
PROD_SEUIL_PAUSE_S = 5.0


def _prod_config_machine():
    nom = st.session_state.get("machine_selection", "Machine 1")
    return nom, MACHINES_CONFIG.get(nom, MACHINES_CONFIG["Machine 1"])


def _prod_reset_commande(fermer_camera=True):
    cam = st.session_state.get("camera_direct")

    if cam is not None and fermer_camera:
        try:
            cam.fermer()
        except Exception:
            pass
        st.session_state.pop("camera_direct", None)

    for cle in (
        "cam_refs",
        "cam_gel",
        "auto_calib_start_num",
        "auto_calib_en_cours",
        "auto_calib_rapport",
        "auto_calib_erreur",
        "auto_global_config",
        "auto_global_processing",
        "wo_scan_input",
        "wo_scan_erreur",
        "prod_terminee",
        "prod_session_id",
        "prod_last_saved_events",
        "prod_last_saved_cables",
        "prod_saved_bilan",
        "prod_saved_events",
        "prod_saved_excel",
    ):
        st.session_state.pop(cle, None)

    st.session_state.work_order = ""
    st.session_state.wo_valide = False
    st.session_state.commande_validee = False


def _prod_paths():
    session_id = st.session_state.get("prod_session_id")

    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["prod_session_id"] = session_id

    machine, _ = _prod_config_machine()
    machine_saine = re.sub(r"[^A-Za-z0-9_-]+", "_", machine)

    wo = st.session_state.get("work_order") or "SANS_WO"
    wo_sain = re.sub(r"[^A-Za-z0-9_-]+", "_", str(wo))
    jour = datetime.now().strftime("%Y-%m-%d")

    dossier = Path("historique_production") / jour
    dossier.mkdir(parents=True, exist_ok=True)

    base_nom = f"{session_id}_{machine_saine}_{wo_sain}"

    return (
        dossier / f"{base_nom}_BILAN.csv",
        dossier / f"{base_nom}_EVENEMENTS.csv",
    )


def _prod_sauvegarder(cam, statut):
    """
    Sauvegarde un snapshot de production sur le PC de la machine.
    Le même fichier est mis à jour pendant la commande puis finalisé à l'arrêt.

    V1.6 :
    les CSV utilisent le séparateur ';' pour une ouverture directe en colonnes
    dans Excel avec les paramètres régionaux français/tunisiens.
    """
    with cam.lock:
        evenements = [dict(e) for e in cam.compteur.evenements]
        cables = int(cam.compteur.cables)
        pauses = int(cam.compteur.pauses_detectees)
        init_ignorees = int(
            cam.compteur.fermetures_ignorees_initialisation
        )
        reprise_ignorees = int(
            cam.compteur.fermetures_ignorees_reprise
        )
        occlusions = int(cam.occlusions_detectees)
        reprises_occlusion = int(
            getattr(cam.compteur, "reprises_forcees_occlusion", 0)
        )
        test_started_at = cam.test_started_at
        test_ended_at = cam.test_ended_at
        images_rejouees = int(
            getattr(cam, "images_rejouees_calibration", 0)
        )
        fermetures_recuperees = int(
            getattr(cam, "fermetures_recuperees_calibration", 0)
        )
        cables_recuperes = int(
            getattr(cam, "cables_recuperes_calibration", 0)
        )

    wo = st.session_state.get("work_order") or ""
    longueur = float(
        st.session_state.get("test_longueur", 0.0)
    )
    quantite = int(
        st.session_state.get("test_quantite_cible", 0)
    )

    restant = max(0, quantite - cables)

    bilan_path, events_path = _prod_paths()

    machine, _ = _prod_config_machine()

    bilan = {
        "Date_snapshot": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Statut": statut,
        "Machine": machine,
        "Work_Order": wo,
        "Longueur_cable_m": longueur,
        "Quantite_prevue": quantite,
        "Cables_detectes": cables,
        "Cables_restants": restant,
        "Fermetures_detectees": len(evenements),
        "Initialisations_ignorees": init_ignorees,
        "Pauses_detectees": pauses,
        "Mouvements_reprise_ignores": reprise_ignorees,
        "Occlusions_detectees": occlusions,
        "Reprises_protegees_apres_occlusion": reprises_occlusion,
        "Images_rejouees_calibration": images_rejouees,
        "Fermetures_recuperees_calibration": fermetures_recuperees,
        "Cables_recuperes_calibration": cables_recuperes,
        "Debut_production": test_started_at or "",
        "Fin_production": test_ended_at or "",
    }

    with bilan_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(bilan.keys()),
            delimiter=";",
            lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(bilan)

    champs = [
        "Machine",
        "Work_Order",
        "Fermeture",
        "Source",
        "Etape_machine",
        "Cable_compte",
        "Pause_avant",
        "Image_debut",
        "Secondes_debut",
        "Image_reouverture",
        "Secondes_reouverture",
        "Image_confirmation",
    ]

    with events_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=champs,
            delimiter=";",
            lineterminator="\n"
        )
        writer.writeheader()

        for evenement in evenements:
            writer.writerow({
                "Machine": machine,
                "Work_Order": wo,
                "Fermeture": evenement.get("Fermeture", ""),
                "Source": evenement.get(
                    "Source",
                    "TEMPS_REEL"
                ),
                "Etape_machine": evenement.get(
                    "Etape_machine",
                    ""
                ),
                "Cable_compte": evenement.get(
                    "Cable_compte",
                    ""
                ),
                "Pause_avant": evenement.get(
                    "Pause_avant",
                    ""
                ),
                "Image_debut": evenement.get(
                    "Image_debut",
                    ""
                ),
                "Secondes_debut": evenement.get(
                    "Secondes_debut",
                    ""
                ),
                "Image_reouverture": evenement.get(
                    "Image_reouverture",
                    ""
                ),
                "Secondes_reouverture": evenement.get(
                    "Secondes_reouverture",
                    ""
                ),
                "Image_confirmation": evenement.get(
                    "Image_confirmation",
                    ""
                ),
            })

    st.session_state["prod_saved_bilan"] = str(bilan_path)
    st.session_state["prod_saved_events"] = str(events_path)

    return bilan_path, events_path


def _prod_generer_excel(cam):
    """
    Génère UN classeur Excel d'analyse complet à la fin de chaque commande.

    Feuilles :
    - RESUME : KPI et durée globale de la commande
    - CABLES : une ligne par câble réellement compté sur F2
    - EVENEMENTS : toutes les fermetures F1/F2/F3
    - DIAGNOSTIC : paramètres et indicateurs techniques

    Le fichier est enregistré dans historique_production\\AAAA-MM-JJ\\.
    """
    try:
        import xlsxwriter
    except ImportError as exc:
        raise ValueError(
            "XlsxWriter n'est pas installé. Exécutez : "
            "python -m pip install XlsxWriter"
        ) from exc

    with cam.lock:
        evenements = [dict(e) for e in cam.compteur.evenements]
        cables = int(cam.compteur.cables)
        pauses = int(cam.compteur.pauses_detectees)
        init_ignorees = int(
            cam.compteur.fermetures_ignorees_initialisation
        )
        reprise_ignorees = int(
            cam.compteur.fermetures_ignorees_reprise
        )
        occlusions = int(cam.occlusions_detectees)
        reprises_occlusion = int(
            getattr(cam.compteur, "reprises_forcees_occlusion", 0)
        )

        test_started_at = cam.test_started_at
        test_ended_at = cam.test_ended_at
        debut_mono = cam.test_started_monotonic
        fin_mono = cam.test_ended_monotonic

        images_rejouees = int(
            getattr(cam, "images_rejouees_calibration", 0)
        )
        fermetures_recuperees = int(
            getattr(cam, "fermetures_recuperees_calibration", 0)
        )
        cables_recuperes = int(
            getattr(cam, "cables_recuperes_calibration", 0)
        )

    machine, profil_machine = _prod_config_machine()
    wo = st.session_state.get("work_order") or "SANS_WO"
    longueur = float(
        st.session_state.get("test_longueur", 0.0)
    )
    quantite_prevue = int(
        st.session_state.get("test_quantite_cible", 0)
    )

    # Heure de début / fin.
    fmt_dt = "%Y-%m-%d %H:%M:%S"

    debut_dt = None
    fin_dt = None

    if test_started_at:
        try:
            debut_dt = datetime.strptime(test_started_at, fmt_dt)
        except ValueError:
            debut_dt = None

    if test_ended_at:
        try:
            fin_dt = datetime.strptime(test_ended_at, fmt_dt)
        except ValueError:
            fin_dt = None

    # Durée précise via monotonic si disponible.
    if debut_mono is not None and fin_mono is not None:
        duree_s = max(0.0, float(fin_mono - debut_mono))
    elif debut_dt is not None and fin_dt is not None:
        duree_s = max(0.0, (fin_dt - debut_dt).total_seconds())
    else:
        duree_s = 0.0

    heures = int(duree_s // 3600)
    minutes = int((duree_s % 3600) // 60)
    secondes = int(duree_s % 60)
    duree_texte = f"{heures:02d}:{minutes:02d}:{secondes:02d}"

    restant = max(0, quantite_prevue - cables)
    ecart = cables - quantite_prevue
    progression = (
        (cables / quantite_prevue) * 100.0
        if quantite_prevue > 0
        else 0.0
    )
    metres_total = cables * longueur
    cadence_h = (
        cables / (duree_s / 3600.0)
        if duree_s > 0
        else 0.0
    )

    # Chemin Excel avec le même identifiant de session.
    session_id = st.session_state.get("prod_session_id")
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["prod_session_id"] = session_id

    machine_saine = re.sub(r"[^A-Za-z0-9_-]+", "_", machine)
    wo_sain = re.sub(r"[^A-Za-z0-9_-]+", "_", str(wo))
    jour = datetime.now().strftime("%Y-%m-%d")
    dossier = Path("historique_production") / jour
    dossier.mkdir(parents=True, exist_ok=True)

    excel_path = dossier / (
        f"{session_id}_{machine_saine}_{wo_sain}_ANALYSE.xlsx"
    )

    workbook = xlsxwriter.Workbook(str(excel_path))

    # --------------------------------------------------------
    # Formats
    # --------------------------------------------------------
    fmt_title = workbook.add_format({
        "bold": True,
        "font_size": 20,
        "align": "center",
        "valign": "vcenter",
        "bg_color": "#1F4E78",
        "font_color": "#FFFFFF",
    })
    fmt_section = workbook.add_format({
        "bold": True,
        "font_size": 12,
        "bg_color": "#D9EAF7",
        "font_color": "#17365D",
        "border": 1,
    })
    fmt_label = workbook.add_format({
        "bold": True,
        "bg_color": "#EAF2F8",
        "border": 1,
    })
    fmt_value = workbook.add_format({
        "border": 1,
    })
    fmt_int = workbook.add_format({
        "border": 1,
        "num_format": "0",
    })
    fmt_decimal = workbook.add_format({
        "border": 1,
        "num_format": "0.00",
    })
    fmt_pct = workbook.add_format({
        "border": 1,
        "num_format": "0.00%",
    })
    fmt_datetime = workbook.add_format({
        "border": 1,
        "num_format": "dd/mm/yyyy hh:mm:ss",
    })
    fmt_time = workbook.add_format({
        "border": 1,
        "num_format": "hh:mm:ss",
    })
    fmt_header = workbook.add_format({
        "bold": True,
        "bg_color": "#1F4E78",
        "font_color": "#FFFFFF",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })
    fmt_cell = workbook.add_format({
        "border": 1,
        "valign": "top",
    })
    fmt_cell_decimal = workbook.add_format({
        "border": 1,
        "num_format": "0.000",
    })
    fmt_good = workbook.add_format({
        "bold": True,
        "bg_color": "#C6EFCE",
        "font_color": "#006100",
        "border": 1,
    })
    fmt_warn = workbook.add_format({
        "bold": True,
        "bg_color": "#FFEB9C",
        "font_color": "#9C6500",
        "border": 1,
    })

    # --------------------------------------------------------
    # 1) RESUME
    # --------------------------------------------------------
    ws = workbook.add_worksheet("RESUME")
    ws.hide_gridlines(2)
    ws.set_column("A:A", 30)
    ws.set_column("B:B", 24)
    ws.set_column("C:C", 3)
    ws.set_column("D:H", 15)

    ws.merge_range("A1:H2", "STARZ — ANALYSE DE COMMANDE", fmt_title)

    ws.write("A4", "IDENTIFICATION", fmt_section)
    ws.write("A5", "Work Order", fmt_label)
    ws.write("B5", wo, fmt_value)
    ws.write("D5", "Machine", fmt_label)
    ws.write("E5", machine, fmt_value)
    ws.write("A6", "Longueur câble (m)", fmt_label)
    ws.write_number("B6", longueur, fmt_decimal)
    ws.write("A7", "Quantité prévue", fmt_label)
    ws.write_number("B7", quantite_prevue, fmt_int)
    ws.write("A8", "Quantité détectée", fmt_label)
    ws.write_number("B8", cables, fmt_int)
    ws.write("A9", "Écart", fmt_label)
    ws.write_number("B9", ecart, fmt_int)
    ws.write("A10", "Restant", fmt_label)
    ws.write_number("B10", restant, fmt_int)
    ws.write("A11", "Progression", fmt_label)
    ws.write_number(
        "B11",
        (cables / quantite_prevue) if quantite_prevue else 0.0,
        fmt_pct
    )
    ws.write("A12", "Mètres produits détectés", fmt_label)
    ws.write_number("B12", metres_total, fmt_decimal)

    ws.write("A14", "TEMPS DE LA COMMANDE", fmt_section)
    ws.write("A15", "Début production", fmt_label)
    if debut_dt:
        ws.write_datetime("B15", debut_dt, fmt_datetime)
    else:
        ws.write("B15", test_started_at or "", fmt_value)

    ws.write("A16", "Fin production", fmt_label)
    if fin_dt:
        ws.write_datetime("B16", fin_dt, fmt_datetime)
    else:
        ws.write("B16", test_ended_at or "", fmt_value)

    ws.write("A17", "Durée globale", fmt_label)
    ws.write("B17", duree_texte, fmt_good)

    ws.write("A18", "Durée globale (secondes)", fmt_label)
    ws.write_number("B18", duree_s, fmt_decimal)

    ws.write("A19", "Cadence moyenne (câbles/h)", fmt_label)
    ws.write_number("B19", cadence_h, fmt_decimal)

    ws.write("A21", "QUALITÉ / DIAGNOSTIC", fmt_section)
    ws.write("A22", "Fermetures détectées", fmt_label)
    ws.write_number("B22", len(evenements), fmt_int)
    ws.write("A23", "Initialisations ignorées", fmt_label)
    ws.write_number("B23", init_ignorees, fmt_int)
    ws.write("A24", "Pauses détectées", fmt_label)
    ws.write_number("B24", pauses, fmt_int)
    ws.write("A25", "Mouvements reprise ignorés", fmt_label)
    ws.write_number("B25", reprise_ignorees, fmt_int)
    ws.write("A26", "Occlusions détectées", fmt_label)
    ws.write_number("B26", occlusions, fmt_int)
    ws.write("D26", "Reprises protégées après occlusion", fmt_label)
    ws.write_number("E26", reprises_occlusion, fmt_int)
    ws.write("A27", "Images rejouées calibration", fmt_label)
    ws.write_number("B27", images_rejouees, fmt_int)
    ws.write("A28", "Fermetures récupérées calibration", fmt_label)
    ws.write_number("B28", fermetures_recuperees, fmt_int)
    ws.write("A29", "Câbles récupérés calibration", fmt_label)
    ws.write_number("B29", cables_recuperes, fmt_int)

    statut_fmt = fmt_good if cables >= quantite_prevue else fmt_warn
    ws.write("A31", "Statut quantité", fmt_label)
    ws.write(
        "B31",
        "OBJECTIF ATTEINT" if cables >= quantite_prevue else "COMMANDE ARRÊTÉE AVANT OBJECTIF",
        statut_fmt
    )

    # --------------------------------------------------------
    # Préparer données événement / câbles.
    # --------------------------------------------------------
    donnees_evenements = []
    donnees_cables = []
    precedente_coupe_s = None

    for evenement in evenements:
        sec_debut = evenement.get("Secondes_debut")
        sec_fin = evenement.get("Secondes_reouverture")

        duree_fermeture = None
        if sec_debut not in (None, "") and sec_fin not in (None, ""):
            try:
                duree_fermeture = float(sec_fin) - float(sec_debut)
            except (TypeError, ValueError):
                duree_fermeture = None

        heure_debut = None
        heure_fin = None

        if debut_dt is not None:
            if sec_debut not in (None, ""):
                try:
                    heure_debut = debut_dt + timedelta(
                        seconds=float(sec_debut)
                    )
                except (TypeError, ValueError):
                    pass

            if sec_fin not in (None, ""):
                try:
                    heure_fin = debut_dt + timedelta(
                        seconds=float(sec_fin)
                    )
                except (TypeError, ValueError):
                    pass

        donnees_evenements.append({
            "Fermeture": evenement.get("Fermeture", ""),
            "Source": evenement.get("Source", "TEMPS_REEL"),
            "Etape_machine": evenement.get("Etape_machine", ""),
            "Cable_compte": evenement.get("Cable_compte", ""),
            "Pause_avant": evenement.get("Pause_avant", ""),
            "Image_debut": evenement.get("Image_debut", ""),
            "Secondes_debut": sec_debut,
            "Heure_debut": heure_debut,
            "Image_reouverture": evenement.get("Image_reouverture", ""),
            "Secondes_reouverture": sec_fin,
            "Heure_reouverture": heure_fin,
            "Duree_fermeture_s": duree_fermeture,
            "Image_confirmation": evenement.get("Image_confirmation", ""),
        })

        if evenement.get("Etape_machine") == "F2_COUPE":
            numero_cable = evenement.get("Cable_compte", "")
            coupe_s = None

            if sec_debut not in (None, ""):
                try:
                    coupe_s = float(sec_debut)
                except (TypeError, ValueError):
                    coupe_s = None

            intervalle_s = None
            if coupe_s is not None and precedente_coupe_s is not None:
                intervalle_s = coupe_s - precedente_coupe_s

            if coupe_s is not None:
                precedente_coupe_s = coupe_s

            heure_coupe = (
                debut_dt + timedelta(seconds=coupe_s)
                if debut_dt is not None and coupe_s is not None
                else None
            )

            try:
                numero_cable_int = int(numero_cable)
            except (TypeError, ValueError):
                numero_cable_int = len(donnees_cables) + 1

            donnees_cables.append({
                "Cable": numero_cable_int,
                "Source": evenement.get("Source", "TEMPS_REEL"),
                "Heure_coupe_estimee": heure_coupe,
                "Temps_depuis_debut_s": coupe_s,
                "Intervalle_depuis_cable_precedent_s": intervalle_s,
                "Longueur_m": longueur,
                "Longueur_cumulee_m": numero_cable_int * longueur,
            })

    # --------------------------------------------------------
    # 2) CABLES
    # --------------------------------------------------------
    ws_c = workbook.add_worksheet("CABLES")
    ws_c.freeze_panes(1, 0)
    ws_c.hide_gridlines(2)

    headers_c = [
        "Cable",
        "Source",
        "Heure coupe estimée",
        "Temps depuis début (s)",
        "Intervalle depuis câble précédent (s)",
        "Longueur (m)",
        "Longueur cumulée (m)",
    ]

    for col, titre in enumerate(headers_c):
        ws_c.write(0, col, titre, fmt_header)

    for row, item in enumerate(donnees_cables, start=1):
        ws_c.write_number(row, 0, item["Cable"], fmt_int)
        ws_c.write(row, 1, item["Source"], fmt_cell)

        if item["Heure_coupe_estimee"] is not None:
            ws_c.write_datetime(
                row, 2, item["Heure_coupe_estimee"], fmt_datetime
            )
        else:
            ws_c.write(row, 2, "", fmt_cell)

        if item["Temps_depuis_debut_s"] is not None:
            ws_c.write_number(
                row, 3, item["Temps_depuis_debut_s"], fmt_cell_decimal
            )
        else:
            ws_c.write(row, 3, "", fmt_cell)

        if item["Intervalle_depuis_cable_precedent_s"] is not None:
            ws_c.write_number(
                row, 4,
                item["Intervalle_depuis_cable_precedent_s"],
                fmt_cell_decimal
            )
        else:
            ws_c.write(row, 4, "", fmt_cell)

        ws_c.write_number(row, 5, item["Longueur_m"], fmt_decimal)
        ws_c.write_number(
            row, 6, item["Longueur_cumulee_m"], fmt_decimal
        )

    ws_c.set_column("A:A", 10)
    ws_c.set_column("B:B", 16)
    ws_c.set_column("C:C", 22)
    ws_c.set_column("D:E", 25)
    ws_c.set_column("F:G", 20)

    if donnees_cables:
        ws_c.add_table(
            0, 0, len(donnees_cables), len(headers_c) - 1,
            {
                "name": "TableCables",
                "columns": [{"header": h} for h in headers_c],
                "style": "Table Style Medium 2",
            }
        )

        # Graphique cadence / temps par câble.
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name": "Intervalle entre câbles (s)",
            "categories": [
                "CABLES", 1, 0, len(donnees_cables), 0
            ],
            "values": [
                "CABLES", 1, 4, len(donnees_cables), 4
            ],
            "line": {"color": "#1F4E78", "width": 2.0},
            "marker": {"type": "circle", "size": 4},
        })
        chart.set_title({"name": "Temps entre deux câbles"})
        chart.set_x_axis({"name": "N° câble"})
        chart.set_y_axis({"name": "Secondes"})
        chart.set_legend({"none": True})
        chart.set_style(10)

        ws.insert_chart("D4", chart, {
            "x_scale": 1.35,
            "y_scale": 1.30,
        })

    # --------------------------------------------------------
    # 3) EVENEMENTS
    # --------------------------------------------------------
    ws_e = workbook.add_worksheet("EVENEMENTS")
    ws_e.freeze_panes(1, 0)
    ws_e.hide_gridlines(2)

    headers_e = [
        "Fermeture",
        "Source",
        "Etape machine",
        "Cable compté",
        "Pause avant",
        "Image début",
        "Secondes début",
        "Heure début",
        "Image réouverture",
        "Secondes réouverture",
        "Heure réouverture",
        "Durée fermeture (s)",
        "Image confirmation",
    ]

    for col, titre in enumerate(headers_e):
        ws_e.write(0, col, titre, fmt_header)

    for row, item in enumerate(donnees_evenements, start=1):
        valeurs_simples = [
            item["Fermeture"],
            item["Source"],
            item["Etape_machine"],
            item["Cable_compte"],
            item["Pause_avant"],
            item["Image_debut"],
        ]

        for col, val in enumerate(valeurs_simples):
            ws_e.write(row, col, val, fmt_cell)

        if item["Secondes_debut"] not in (None, ""):
            ws_e.write_number(
                row, 6, float(item["Secondes_debut"]), fmt_cell_decimal
            )
        else:
            ws_e.write(row, 6, "", fmt_cell)

        if item["Heure_debut"] is not None:
            ws_e.write_datetime(
                row, 7, item["Heure_debut"], fmt_datetime
            )
        else:
            ws_e.write(row, 7, "", fmt_cell)

        ws_e.write(row, 8, item["Image_reouverture"], fmt_cell)

        if item["Secondes_reouverture"] not in (None, ""):
            ws_e.write_number(
                row, 9,
                float(item["Secondes_reouverture"]),
                fmt_cell_decimal
            )
        else:
            ws_e.write(row, 9, "", fmt_cell)

        if item["Heure_reouverture"] is not None:
            ws_e.write_datetime(
                row, 10, item["Heure_reouverture"], fmt_datetime
            )
        else:
            ws_e.write(row, 10, "", fmt_cell)

        if item["Duree_fermeture_s"] is not None:
            ws_e.write_number(
                row, 11, item["Duree_fermeture_s"], fmt_cell_decimal
            )
        else:
            ws_e.write(row, 11, "", fmt_cell)

        ws_e.write(row, 12, item["Image_confirmation"], fmt_cell)

    ws_e.set_column("A:A", 12)
    ws_e.set_column("B:B", 16)
    ws_e.set_column("C:C", 28)
    ws_e.set_column("D:E", 14)
    ws_e.set_column("F:G", 17)
    ws_e.set_column("H:H", 22)
    ws_e.set_column("I:J", 20)
    ws_e.set_column("K:K", 22)
    ws_e.set_column("L:M", 20)

    if donnees_evenements:
        ws_e.add_table(
            0, 0, len(donnees_evenements), len(headers_e) - 1,
            {
                "name": "TableEvenements",
                "columns": [{"header": h} for h in headers_e],
                "style": "Table Style Medium 2",
            }
        )

    # --------------------------------------------------------
    # 4) DIAGNOSTIC
    # --------------------------------------------------------
    ws_d = workbook.add_worksheet("DIAGNOSTIC")
    ws_d.hide_gridlines(2)
    ws_d.set_column("A:A", 34)
    ws_d.set_column("B:B", 26)

    diagnostic = [
        ("Machine", machine),
        ("Work Order", wo),
        ("Longueur câble (m)", longueur),
        ("Quantité prévue", quantite_prevue),
        ("Quantité détectée", cables),
        ("Durée globale", duree_texte),
        ("Cadence moyenne câbles/h", round(cadence_h, 3)),
        ("Fermetures détectées", len(evenements)),
        ("Initialisations ignorées", init_ignorees),
        ("Pauses détectées", pauses),
        ("Mouvements reprise ignorés", reprise_ignorees),
        ("Occlusions détectées", occlusions),
        ("Reprises protégées après occlusion", reprises_occlusion),
        ("Images rejouées calibration", images_rejouees),
        ("Fermetures récupérées calibration", fermetures_recuperees),
        ("Câbles récupérés calibration", cables_recuperes),
        ("Caméra index", profil_machine["camera_index"]),
        ("Rotation caméra", profil_machine["rotation"]),
        ("ROI gauche %", profil_machine["roi"][0]),
        ("ROI droite %", profil_machine["roi"][1]),
        ("ROI haut %", profil_machine["roi"][2]),
        ("ROI bas %", profil_machine["roi"][3]),
        ("Calibration images", PROD_AUTO_IMAGES),
        ("Références par état", PROD_NB_REFS),
        ("Marge", PROD_MARGE),
        ("Seuil différence", PROD_SEUIL),
        ("Confirmation images", PROD_CONFIRMATION),
        ("Seuil pause temps réel (s)", PROD_SEUIL_PAUSE_S),
    ]

    ws_d.write("A1", "Paramètre / indicateur", fmt_header)
    ws_d.write("B1", "Valeur", fmt_header)

    for row, (nom, valeur) in enumerate(diagnostic, start=1):
        ws_d.write(row, 0, nom, fmt_label)
        ws_d.write(row, 1, valeur, fmt_value)

    workbook.close()

    st.session_state["prod_saved_excel"] = str(excel_path)
    return excel_path


def afficher_camera_production():
    """
    Interface finale destinée à l'opérateur :
    - scanner WO facultatif ;
    - longueur + quantité ;
    - connexion caméra automatique ;
    - un seul bouton Démarrer ;
    - calibration/replay invisibles pour l'opérateur ;
    - compteur global simple ;
    - sauvegarde automatique des résultats.
    """
    if not hasattr(st, "fragment"):
        st.error(
            "Streamlit doit être mis à jour avant utilisation."
        )
        return

    # --------------------------------------------------------
    # STYLE PRODUCTION
    # --------------------------------------------------------
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .starz-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .starz-subtitle {
            font-size: 1.0rem;
            opacity: 0.72;
            margin-bottom: 1.5rem;
        }
        .prod-card {
            border: 1px solid rgba(128,128,128,.28);
            border-radius: 16px;
            padding: 18px 22px;
            margin: 8px 0 18px 0;
        }
        .prod-big {
            font-size: 4.4rem;
            line-height: 1;
            font-weight: 900;
            text-align: center;
            margin: 0.2rem 0;
        }
        .prod-label {
            text-align: center;
            font-size: 1.05rem;
            opacity: .75;
        }
        div[data-testid="stTextInput"]:has(
            input[aria-label="WO_PROD_HIDDEN"]
        ) {
            position: fixed !important;
            left: -10000px !important;
            top: -10000px !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="starz-title">STARZ — Compteur de câbles</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="starz-subtitle">Poste opérateur de production</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INITIALISATION SESSION
    # --------------------------------------------------------
    if "wo_valide" not in st.session_state:
        st.session_state.wo_valide = False

    if "commande_validee" not in st.session_state:
        st.session_state.commande_validee = False

    if "work_order" not in st.session_state:
        st.session_state.work_order = ""

    if "machine_validee" not in st.session_state:
        st.session_state.machine_validee = False

    if "machine_selection" not in st.session_state:
        st.session_state.machine_selection = "Machine 1"

    # --------------------------------------------------------
    # ETAPE 0 — CHOIX MACHINE
    # --------------------------------------------------------
    if not st.session_state.machine_validee:
        st.subheader("Choisir la machine")
        st.write(
            "Sélectionnez la machine utilisée. "
            "La caméra et la zone de détection seront réglées automatiquement."
        )

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            if st.button(
                "Machine 1",
                type="primary",
                use_container_width=True,
                key="choisir_machine_1"
            ):
                st.session_state.machine_selection = "Machine 1"
                st.session_state.machine_validee = True
                st.rerun()

        with col_m2:
            if st.button(
                "Machine 2",
                type="primary",
                use_container_width=True,
                key="choisir_machine_2"
            ):
                st.session_state.machine_selection = "Machine 2"
                st.session_state.machine_validee = True
                st.rerun()

        return

    machine_nom, profil_machine = _prod_config_machine()
    st.caption(f"Machine sélectionnée : {machine_nom}")

    # --------------------------------------------------------
    # ETAPE 1 — SCAN WO FACULTATIF, SANS CHAMP VISIBLE
    # --------------------------------------------------------
    if not st.session_state.wo_valide:
        st.subheader("Scanner le Work Order")

        st.info(
            "Scannez le Work Order directement avec le lecteur. "
            "Aucun clic n'est nécessaire."
        )

        def _prod_traiter_scan():
            brut = st.session_state.get(
                "wo_scan_input",
                ""
            )
            wo = normaliser_work_order_scanner(brut)

            if wo is None:
                st.session_state["wo_scan_erreur"] = (
                    "Code non reconnu. Rescannez le Work Order."
                )
                return

            st.session_state.work_order = wo
            st.session_state.wo_valide = True
            st.session_state.pop("wo_scan_erreur", None)

        st.text_input(
            "WO_PROD_HIDDEN",
            key="wo_scan_input",
            on_change=_prod_traiter_scan,
            label_visibility="collapsed"
        )

        components.html(
            """
            <script>
            (function () {
                function focusScanner() {
                    try {
                        const doc = window.parent.document;
                        const el = doc.querySelector(
                            'input[aria-label="WO_PROD_HIDDEN"]'
                        );
                        if (el) {
                            el.focus();
                            el.setAttribute('autocomplete', 'off');
                            return true;
                        }
                    } catch (e) {}
                    return false;
                }
                let n = 0;
                const timer = setInterval(() => {
                    n += 1;
                    if (focusScanner() || n >= 40) {
                        clearInterval(timer);
                    }
                }, 100);
                setTimeout(focusScanner, 50);
            })();
            </script>
            """,
            height=0,
            width=0
        )

        erreur_scan = st.session_state.get(
            "wo_scan_erreur"
        )

        if erreur_scan:
            st.error(erreur_scan)

        if st.button(
            "Continuer sans Work Order",
            use_container_width=True
        ):
            st.session_state.work_order = ""
            st.session_state.wo_valide = True
            st.session_state.pop("wo_scan_erreur", None)
            st.rerun()

        if st.button(
            "Changer de machine",
            use_container_width=True
        ):
            st.session_state.machine_validee = False
            st.session_state.wo_valide = False
            st.session_state.work_order = ""
            st.session_state.pop("wo_scan_input", None)
            st.session_state.pop("wo_scan_erreur", None)
            st.rerun()

        return

    # --------------------------------------------------------
    # ETAPE 2 — LONGUEUR + QUANTITE
    # --------------------------------------------------------
    if not st.session_state.commande_validee:
        wo = st.session_state.work_order

        if wo:
            st.success(f"Work Order : {wo}")
        else:
            st.info("Production sans Work Order")

        st.subheader("Informations de la commande")

        with st.form("prod_commande_form"):
            c1, c2 = st.columns(2)

            with c1:
                longueur = st.number_input(
                    "Longueur du câble (m)",
                    min_value=0.01,
                    value=1.10,
                    step=0.10,
                    format="%.2f"
                )

            with c2:
                quantite = st.number_input(
                    "Quantité à produire",
                    min_value=1,
                    value=20,
                    step=1
                )

            valider = st.form_submit_button(
                "Valider la commande",
                type="primary",
                use_container_width=True
            )

        if valider:
            st.session_state.test_id = (
                wo if wo else "SANS_WO"
            )
            st.session_state.test_longueur = float(longueur)
            st.session_state.test_quantite_cible = int(quantite)
            st.session_state.test_commentaire = ""
            st.session_state.commande_validee = True
            st.session_state["prod_terminee"] = False
            st.rerun()

        return

    # --------------------------------------------------------
    # CONNEXION AUTOMATIQUE CAMERA
    # --------------------------------------------------------
    cam = st.session_state.get("camera_direct")

    if cam is None:
        try:
            machine_nom, profil_machine = _prod_config_machine()

            cam = Camera(
                profil_machine["camera_index"],
                profil_machine["roi"],
                rotation=profil_machine["rotation"],
                activer_barcode=False,
                verrouiller_exposition=(
                    profil_machine["verrouiller_exposition"]
                )
            )

            cam.code_barres = (
                st.session_state.work_order or None
            )
            cam.code_barres_format = (
                "SCANNER_USB"
                if st.session_state.work_order
                else None
            )
            cam.scan_barcode_actif = False

            st.session_state.camera_direct = cam
            st.session_state.cam_refs = {
                "ouvert": [],
                "ferme": []
            }

            st.session_state.pop(
                "auto_calib_en_cours",
                None
            )
            st.session_state.pop(
                "auto_calib_rapport",
                None
            )
            st.session_state.pop(
                "auto_calib_erreur",
                None
            )
            st.session_state.pop(
                "auto_global_processing",
                None
            )

            st.rerun()

        except Exception as exc:
            st.error(
                f"Connexion caméra impossible : {exc}"
            )

            if st.button("Réessayer"):
                st.rerun()

            return

    # --------------------------------------------------------
    # PANNEAU OPERATEUR
    # --------------------------------------------------------
    @st.fragment(run_every=0.25)
    def panneau_production():
        cam.heartbeat = time.monotonic()

        with cam.lock:
            actif = bool(cam.actif)
            erreur = cam.erreur
            cables = int(cam.compteur.cables)
            evenements = list(
                cam.compteur.evenements
            )
            camera_pret = cam.derniere is not None
            etat = cam.etat

        wo = st.session_state.work_order
        longueur = float(
            st.session_state.test_longueur
        )
        objectif = int(
            st.session_state.test_quantite_cible
        )

        # En-tête commande
        machine_nom, _ = _prod_config_machine()
        c0, c1, c2, c3 = st.columns(4)
        c0.metric(
            "Machine",
            machine_nom
        )
        c1.metric(
            "Work Order",
            wo if wo else "Sans WO"
        )
        c2.metric(
            "Longueur",
            f"{longueur:.2f} m"
        )
        c3.metric(
            "Objectif",
            objectif
        )

        if erreur:
            st.error(
                f"Erreur caméra / compteur : {erreur}"
            )

            if st.button(
                "Réinitialiser le poste",
                type="primary",
                use_container_width=True
            ):
                _prod_reset_commande(
                    fermer_camera=True
                )
                st.rerun()

            return

        # ----------------------------------------------------
        # COMMANDE TERMINEE
        # ----------------------------------------------------
        if st.session_state.get(
            "prod_terminee",
            False
        ):
            st.success("Commande terminée")

            st.markdown(
                f'<div class="prod-big">{cables}</div>'
                '<div class="prod-label">'
                'câbles comptés'
                '</div>',
                unsafe_allow_html=True
            )

            bilan = st.session_state.get(
                "prod_saved_bilan"
            )
            excel = st.session_state.get(
                "prod_saved_excel"
            )

            with cam.lock:
                debut_aff = cam.test_started_at
                fin_aff = cam.test_ended_at
                debut_mono_aff = cam.test_started_monotonic
                fin_mono_aff = cam.test_ended_monotonic

            if (
                debut_mono_aff is not None
                and fin_mono_aff is not None
            ):
                duree_aff_s = max(
                    0.0,
                    fin_mono_aff - debut_mono_aff
                )
                hh = int(duree_aff_s // 3600)
                mm = int((duree_aff_s % 3600) // 60)
                ss = int(duree_aff_s % 60)

                st.info(
                    f"Début : {debut_aff or '-'}  |  "
                    f"Fin : {fin_aff or '-'}  |  "
                    f"Durée globale : {hh:02d}:{mm:02d}:{ss:02d}"
                )

            if bilan:
                st.caption(
                    "Résultats sauvegardés automatiquement "
                    "sur le PC de la machine."
                )

            if excel and Path(excel).exists():
                st.success(
                    "Excel d'analyse de la commande généré automatiquement."
                )

                with open(excel, "rb") as fichier_excel:
                    st.download_button(
                        "📊 Télécharger Excel d'analyse",
                        data=fichier_excel.read(),
                        file_name=Path(excel).name,
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                        use_container_width=True,
                    )

                st.caption(f"Fichier local : {excel}")

            if st.button(
                "Nouvelle commande",
                type="primary",
                use_container_width=True
            ):
                _prod_reset_commande(
                    fermer_camera=True
                )
                st.rerun()

            return

        # ----------------------------------------------------
        # AVANT DEMARRAGE
        # ----------------------------------------------------
        auto_en_cours = bool(
            st.session_state.get(
                "auto_calib_en_cours",
                False
            )
        )

        if not actif and not auto_en_cours:
            if not camera_pret:
                st.info(
                    "Initialisation de la caméra…"
                )
                return

            st.success("Caméra prête")

            st.markdown(
                """
                <div class="prod-card">
                La commande est prête. Appuyez sur
                <b>Démarrer la production</b>, puis utilisez
                la machine normalement. L'initialisation du
                compteur est automatique.
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "▶ Démarrer la production",
                type="primary",
                use_container_width=True
            ):
                try:
                    refs = st.session_state.cam_refs
                    refs["ouvert"].clear()
                    refs["ferme"].clear()

                    st.session_state[
                        "prod_session_id"
                    ] = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    st.session_state[
                        "prod_last_saved_events"
                    ] = -1
                    st.session_state[
                        "prod_last_saved_cables"
                    ] = -1

                    cam.demarrer_capture_calibration()

                    st.session_state[
                        "auto_calib_en_cours"
                    ] = True
                    st.session_state[
                        "auto_global_processing"
                    ] = False
                    st.session_state.pop(
                        "auto_calib_rapport",
                        None
                    )
                    st.session_state.pop(
                        "auto_calib_erreur",
                        None
                    )

                    st.rerun(
                        scope="fragment"
                    )

                except Exception as exc:
                    st.error(
                        f"Démarrage impossible : {exc}"
                    )

            if st.button(
                "Changer de commande",
                use_container_width=True
            ):
                _prod_reset_commande(
                    fermer_camera=True
                )
                st.rerun()

            return

        # ----------------------------------------------------
        # INITIALISATION AUTOMATIQUE 800 IMAGES
        # ----------------------------------------------------
        if auto_en_cours:
            images_calibration = (
                cam.obtenir_images_calibration()
            )
            nb = len(images_calibration)

            progression = min(
                1.0,
                nb / float(PROD_AUTO_IMAGES)
            )

            st.subheader(
                "Initialisation automatique du compteur"
            )
            st.progress(progression)

            st.write(
                f"Préparation : "
                f"{min(nb, PROD_AUTO_IMAGES)} / "
                f"{PROD_AUTO_IMAGES}"
            )

            st.info(
                "Continuez la production normalement. "
                "Les câbles coupés maintenant seront "
                "récupérés automatiquement."
            )

            if (
                nb >= PROD_AUTO_IMAGES
                and not st.session_state.get(
                    "auto_global_processing",
                    False
                )
            ):
                st.session_state[
                    "auto_global_processing"
                ] = True

                try:
                    refs = st.session_state.cam_refs

                    refs_auto, rapport = (
                        calibrer_references_automatiques(
                            images_calibration[
                                :PROD_AUTO_IMAGES
                            ],
                            nb_refs=PROD_NB_REFS,
                            nb_images_ouvert_initial=15
                        )
                    )

                    refs["ouvert"].clear()
                    refs["ferme"].clear()
                    refs["ouvert"].extend(
                        refs_auto["ouvert"]
                    )
                    refs["ferme"].extend(
                        refs_auto["ferme"]
                    )

                    test_meta = {
                        "Test_ID": (
                            wo if wo else "SANS_WO"
                        ),
                        "Quantite_cible_cables": (
                            objectif
                        ),
                        "Longueur_cable_m": longueur,
                        "Commentaire": (
                            "Interface Production V1"
                        ),
                    }

                    cam.demarrer(
                        refs,
                        PROD_MARGE,
                        PROD_SEUIL,
                        PROD_CONFIRMATION,
                        test_meta=test_meta,
                        ignorer_initiales=2,
                        seuil_pause_s=(
                            PROD_SEUIL_PAUSE_S
                        ),
                        recuperer_calibration=True
                    )

                    st.session_state[
                        "auto_calib_rapport"
                    ] = rapport
                    st.session_state[
                        "auto_calib_en_cours"
                    ] = False
                    st.session_state[
                        "auto_global_processing"
                    ] = False

                    _prod_sauvegarder(
                        cam,
                        "EN_COURS"
                    )

                    st.rerun(
                        scope="fragment"
                    )

                except Exception as exc:
                    cam.annuler_capture_calibration()

                    st.session_state[
                        "auto_calib_en_cours"
                    ] = False
                    st.session_state[
                        "auto_global_processing"
                    ] = False

                    st.error(
                        "Initialisation automatique "
                        f"impossible : {exc}"
                    )

            return

        # ----------------------------------------------------
        # PRODUCTION EN COURS
        # ----------------------------------------------------
        if actif:
            restant = max(
                0,
                objectif - cables
            )

            progression = (
                min(1.0, cables / objectif)
                if objectif > 0
                else 0.0
            )

            st.markdown(
                f'<div class="prod-big">{cables}</div>'
                '<div class="prod-label">'
                'câbles produits'
                '</div>',
                unsafe_allow_html=True
            )

            st.progress(progression)

            c4, c5 = st.columns(2)
            c4.metric(
                "Restants",
                restant
            )
            c5.metric(
                "Progression",
                f"{progression * 100:.1f} %"
            )

            if cables >= objectif:
                st.success(
                    "Objectif atteint — la quantité prévue "
                    "a été comptée."
                )
            else:
                st.caption(
                    "Production en cours — "
                    f"état compteur : {etat}"
                )

            # Sauvegarde automatique uniquement lorsqu'un
            # événement ou le nombre de câbles change.
            nb_events = len(evenements)

            if (
                nb_events
                != st.session_state.get(
                    "prod_last_saved_events",
                    -1
                )
                or cables
                != st.session_state.get(
                    "prod_last_saved_cables",
                    -1
                )
            ):
                try:
                    _prod_sauvegarder(
                        cam,
                        "EN_COURS"
                    )

                    st.session_state[
                        "prod_last_saved_events"
                    ] = nb_events
                    st.session_state[
                        "prod_last_saved_cables"
                    ] = cables

                except Exception:
                    # Le comptage ne doit jamais être bloqué
                    # uniquement par un problème de sauvegarde.
                    pass

            if st.button(
                "■ Terminer la commande",
                type="primary",
                use_container_width=True
            ):
                cam.arreter()

                erreurs_sauvegarde = []

                try:
                    _prod_sauvegarder(
                        cam,
                        "TERMINEE"
                    )
                except Exception as exc:
                    erreurs_sauvegarde.append(
                        f"CSV : {exc}"
                    )

                try:
                    _prod_generer_excel(cam)
                except Exception as exc:
                    erreurs_sauvegarde.append(
                        f"Excel : {exc}"
                    )

                if erreurs_sauvegarde:
                    st.warning(
                        "Commande arrêtée mais certaines sauvegardes "
                        "ont échoué : "
                        + " | ".join(erreurs_sauvegarde)
                    )

                st.session_state[
                    "prod_terminee"
                ] = True

                st.rerun(
                    scope="fragment"
                )

    panneau_production()
