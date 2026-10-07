"""Caméra locale : acquisition, lecture OF par CODE_128 et comptage continu."""
import csv
import io
import platform
import re
import threading
import time
from collections import deque
from datetime import datetime

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
    Normalise la saisie provenant d'un scanner USB qui se comporte comme
    un clavier.

    Exemples :
      48225-1          -> 48225-1
      48225-148225-1   -> 48225-1 si le même code a été scanné 2 fois
      deux WO différents concaténés -> refus
    """
    brut = (texte or "").strip().replace("\r", "").replace("\n", "")

    if FORMAT_OF.fullmatch(brut):
        return brut

    correspondances = re.findall(r"\d{5}-\d", brut)

    if correspondances and len(set(correspondances)) == 1:
        return correspondances[0]

    return None


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

    def reprendre_apres_occlusion(self):
        """Repart proprement depuis OUVERT sans créer une fausse fermeture."""
        self.stable = 'OUVERT'
        self.candidat = None
        self.repetitions = 0
        self.debut = None



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

                if time.monotonic() - self.heartbeat > 30:
                    raise ValueError(
                        'Connexion à la page interrompue : caméra arrêtée.'
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
                                self.compteur.reprendre_apres_occlusion()

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
