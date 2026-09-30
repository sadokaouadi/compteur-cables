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
import zxingcpp

from analyse import preparer_references, estimer_etat


# ============================================================
# CODE-BARRES / OF
# ============================================================

# Format actuellement attendu dans vos feuilles :
# 48247-1, 48246-1, etc.
FORMAT_OF = re.compile(r"^\d{5}-\d$")


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
        pause_avant = False

        if self.derniere_fermeture_s is not None:
            ecart = maintenant - self.derniere_fermeture_s

            # On resynchronise seulement si on était au milieu d'un cycle.
            # Si phase == 0, une longue attente avant F1 ne change rien.
            if (
                ecart >= self.seuil_pause_s
                and self.phase != 0
            ):
                self.pauses_detectees += 1
                self.phase = 0
                pause_avant = True

        evenement["Pause_avant"] = "OUI" if pause_avant else "NON"

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
        if etat == 'INDETERMINE':
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
        self.historique = deque(maxlen=600)

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
            cap.set(cv2.CAP_PROP_FPS, 30)

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

                    # ==================================================
                    # 3) ANALYSE DU MOUVEMENT SI COMPTAGE ACTIF
                    # ==================================================
                    if self.actif:
                        gris = cv2.cvtColor(
                            zone,
                            cv2.COLOR_RGB2GRAY
                        )

                        self.etat, _, _ = estimer_etat(
                            gris,
                            self.refs,
                            self.marge,
                            self.seuil
                        )

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

    def demarrer(
        self,
        refs,
        marge,
        seuil,
        confirmation,
        test_meta=None,
        ignorer_initiales=2,
        seuil_pause_s=5.0
    ):
        with self.lock:

            if self.erreur or not self.thread.is_alive():
                raise ValueError(
                    'Reconnectez la caméra avant de démarrer.'
                )

            self.refs = preparer_references(refs)

            if any(
                (a == b).all()
                for a in self.refs['ouvert']
                for b in self.refs['ferme']
            ):
                raise ValueError(
                    'Les références ouverte et fermée sont identiques.'
                )

            # Chaque démarrage crée un nouveau comptage avec la logique
            # réelle F1 / F2 / F3 de la machine.
            self.compteur = Compteur(
                confirmation=confirmation,
                ignorer_initiales=ignorer_initiales,
                seuil_pause_s=seuil_pause_s
            )
            self.marge = marge
            self.seuil = seuil
            self.confirmation_test = confirmation
            self.debut_test = time.monotonic()

            # On fige les informations du test au moment du démarrage.
            self.test_meta = dict(test_meta or {})
            self.test_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.test_ended_at = None
            self.test_started_monotonic = time.monotonic()
            self.test_ended_monotonic = None

            # L'OF est facultatif pendant le test machine.
            # S'il a été scanné, on le mémorise ; sinon None.
            self.of_compteur = self.code_barres

            # Pendant le comptage, on ne change pas d'OF.
            self.scan_barcode_actif = False

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

    activer_barcode = st.checkbox(
        'Activer la lecture du code-barres / OF (facultatif)',
        value=False,
        disabled=connectee
    )

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

            st.session_state.cam_refs = {
                'ouvert': [],
                'ferme': []
            }

            st.session_state.pop(
                'cam_gel',
                None
            )

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
    # FICHE DU TEST DE VALIDATION
    # ========================================================

    st.subheader("Test de validation")

    if "test_id" not in st.session_state:
        st.session_state.test_id = "TEST_001"
    if "test_quantite_cible" not in st.session_state:
        st.session_state.test_quantite_cible = 1
    if "test_longueur" not in st.session_state:
        st.session_state.test_longueur = 1.0
    if "test_commentaire" not in st.session_state:
        st.session_state.test_commentaire = ""

    c1, c2 = st.columns(2)

    with c1:
        st.text_input(
            "ID du test",
            key="test_id"
        )

        st.number_input(
            "Quantité cible de câbles",
            min_value=1,
            step=1,
            key="test_quantite_cible"
        )

    with c2:
        st.number_input(
            "Longueur du câble (m)",
            min_value=0.0,
            step=0.1,
            key="test_longueur"
        )

    st.text_input(
        "Commentaire du test (optionnel)",
        key="test_commentaire",
        placeholder="Ex. production normale, éclairage atelier, test après fixation..."
    )

    st.caption(
        "Ces informations servent uniquement à la traçabilité et au bilan. "
        "Elles ne modifient pas l'algorithme de comptage."
    )

    if st.button("Nouveau test — remettre le résultat à zéro"):
        try:
            test_id_actuel = st.session_state.get("test_id", "TEST_001")
            cam.nouveau_test()
            st.session_state.pop(
                f"test_quantite_reelle_{test_id_actuel}",
                None
            )
            st.success(
                "Résultats remis à zéro. Les références OUVERT/FERMÉ sont conservées."
            )
        except ValueError as exc:
            st.error(str(exc))

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

        st.caption(
            f'{numero} images reçues. '
            f'Cadence moyenne reçue : {cadence:.1f} images/s. '
            f'Affichage ralenti, analyse de chaque image reçue.'
        )

        # ====================================================
        # CODE-BARRES / OF
        # ====================================================

        st.subheader(
            'Code-barres / OF'
        )

        if code_barres:
            st.success(
                f'OF détecté : {code_barres}'
            )

            st.caption(
                f'Format : {code_barres_format}'
            )

        else:
            if scan_barcode_actif:
                st.info(
                    'Lecture OF active mais facultative : vous pouvez scanner '
                    'une feuille ou continuer le test sans OF.'
                )
            else:
                st.info(
                    "OF non scanné — ce n'est pas obligatoire pour le test machine."
                )

        if (
            not actif
            and not code_barres
            and not scan_barcode_actif
        ):
            if st.button('Scanner un OF maintenant (facultatif)'):
                with cam.lock:
                    cam.scan_barcode_actif = True
                st.rerun(scope='fragment')

        if (
            not actif
            and code_barres
        ):
            if st.button(
                'Scanner un nouvel OF'
            ):
                try:
                    cam.nouveau_of()
                    st.rerun(
                        scope='fragment'
                    )

                except ValueError as exc:
                    st.error(
                        str(exc)
                    )

        refs = st.session_state.cam_refs

        # ====================================================
        # PARAMETRAGE AVANT COMPTAGE
        # ====================================================

        if not actif:

            st.write(
                'Références : filmez quelques mouvements, '
                'puis figez les images récentes pour sélectionner '
                'ouvert et fermé.'
            )

            if st.button(
                'Figer les images récentes',
                disabled=bool(erreur)
            ):
                with cam.lock:
                    st.session_state.cam_gel = list(
                        cam.historique
                    )

                st.session_state.pop(
                    'cam_selection',
                    None
                )

            gel = st.session_state.get(
                'cam_gel',
                []
            )

            if gel:

                selection = st.select_slider(
                    'Image à classer',
                    options=list(range(len(gel))),
                    key='cam_selection'
                )

                n, z = gel[selection]

                st.image(
                    z,
                    width=280,
                    caption=f'Image {n} — sélection figée'
                )

                for nom, libelle in [
                    ('ouvert', 'OUVERT'),
                    ('ferme', 'FERMÉ')
                ]:

                    if st.button(
                        f'Ajouter comme {libelle}'
                    ):

                        deja_classee = any(
                            r['numero'] == n
                            for valeurs in refs.values()
                            for r in valeurs
                        )

                        if deja_classee:
                            st.warning(
                                'Cette image est déjà classée. '
                                'Effacez les références pour corriger.'
                            )
                        else:
                            refs[nom].append(
                                {
                                    'numero': n,
                                    'image': z.copy()
                                }
                            )

            st.caption(
                f"Ouvert : {len(refs['ouvert'])} référence(s) ; "
                f"fermé : {len(refs['ferme'])} référence(s)."
            )

            if st.button(
                'Effacer les références'
            ):
                refs['ouvert'].clear()
                refs['ferme'].clear()

            marge = st.number_input(
                'Marge minimale',
                0.0,
                20.0,
                0.5,
                0.1
            )

            seuil = st.number_input(
                'Différence maximale',
                1.0,
                100.0,
                40.0,
                1.0
            )

            confirmation = st.number_input(
                'Images de confirmation',
                1,
                10,
                1
            )

            nouvelle_commande = st.checkbox(
                "Début d'une nouvelle commande : ignorer les 2 fermetures d'initialisation",
                value=True,
                help=(
                    "À cocher lorsque le comptage démarre au début d'une nouvelle commande. "
                    "Les 2 premiers mouvements F2 + F3 de mise en route ne produisent pas "
                    "encore un câble et seront ignorés."
                )
            )

            seuil_pause_s = st.number_input(
                "Pause / resynchronisation après (secondes)",
                min_value=2.0,
                max_value=120.0,
                value=5.0,
                step=1.0,
                help=(
                    "Si la machine s'arrête au milieu d'un cycle pendant au moins ce temps, "
                    "la prochaine fermeture est traitée comme un nouveau F1."
                )
            )

            st.caption(
                'OF facultatif pendant le test machine. '
                'Chaque démarrage remet le compteur à zéro. '
                'Démarrez avant un cycle complet de production. '
                'Le câble est maintenant compté sur F2 (coupe réelle), '
                'et non plus par simple groupe de 3 fermetures.'
            )

            demarrage_desactive = (
                bool(erreur)
                or not all(refs.values())
            )

            if st.button(
                'Démarrer le comptage',
                disabled=demarrage_desactive
            ):
                try:
                    test_meta = {
                        'Test_ID': st.session_state.get('test_id', 'TEST_001'),
                        'Quantite_cible_cables': int(
                            st.session_state.get('test_quantite_cible', 1)
                        ),
                        'Longueur_cable_m': float(
                            st.session_state.get('test_longueur', 0.0)
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
                        ignorer_initiales=(2 if nouvelle_commande else 0),
                        seuil_pause_s=seuil_pause_s
                    )

                    st.rerun(
                        scope='fragment'
                    )

                except ValueError as exc:
                    st.error(
                        str(exc)
                    )

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

        st.metric(
            'Fermetures complètes',
            len(evenements)
        )

        st.metric(
            'Câbles comptés — coupe F2',
            cables_comptes
        )

        noms_phase = {
            0: "F1 attendu — traçage",
            1: "F2 attendu — coupe",
            2: "F3 attendu — préparation suivant",
        }

        st.caption(
            f"Synchronisation machine : {noms_phase.get(phase_machine, 'inconnue')} — "
            f"pauses/resynchronisations détectées : {pauses_detectees} — "
            f"fermetures d'initialisation ignorées : {initiales_ignorees}."
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
                f"**Initialisation ignorée :** {initiales_ignorees} fermeture(s)  |  "
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
