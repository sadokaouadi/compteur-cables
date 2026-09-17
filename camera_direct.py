"""Caméra locale : acquisition, lecture OF par CODE_128 et comptage continu."""
import csv
import io
import platform
import re
import threading
import time
from collections import deque

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
    def __init__(self, confirmation=1):
        self.confirmation = confirmation
        self.stable = None
        self.candidat = None
        self.debut = None
        self.repetitions = 0
        self.evenements = []
        self.initiale = False

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
            self.evenements.append(
                {
                    'Fermeture': len(self.evenements) + 1,
                    'Image_debut': self.debut[0],
                    'Secondes_debut': round(self.debut[1], 3),
                    'Image_reouverture': self.premier[0],
                    'Secondes_reouverture': round(self.premier[1], 3),
                    'Image_confirmation': numero,
                }
            )
            self.debut = None


# ============================================================
# CAMERA
# ============================================================

class Camera:
    def __init__(self, index, limites):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        self.limites = limites
        self.derniere = None
        self.historique = deque(maxlen=180)

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

        # True = recherche automatique d'un OF
        self.scan_barcode_actif = True

        # OF verrouillé au moment où le comptage démarre
        self.of_compteur = None

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

    def demarrer(self, refs, marge, seuil, confirmation):
        with self.lock:

            if self.erreur or not self.thread.is_alive():
                raise ValueError(
                    'Reconnectez la caméra avant de démarrer.'
                )

            if not self.code_barres:
                raise ValueError(
                    "Aucun OF détecté. "
                    "Présentez d'abord le code-barres à la caméra."
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

            self.compteur = Compteur(confirmation)
            self.marge = marge
            self.seuil = seuil
            self.debut_test = time.monotonic()

            # Verrouille l'OF utilisé pour cette session de comptage.
            self.of_compteur = self.code_barres

            # Pendant le comptage, on ne change pas d'OF.
            self.scan_barcode_actif = False

            self.actif = True

    def arreter(self):
        with self.lock:
            self.actif = False

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

    gx, dx = st.slider(
        'Zone horizontale (%)',
        0,
        100,
        (44, 54),
        disabled=connectee
    )

    hy, by = st.slider(
        'Zone verticale (%)',
        0,
        100,
        (53, 65),
        disabled=connectee
    )

    st.caption(
        'Pour modifier la zone ou changer de caméra, '
        'déconnectez puis reconnectez. '
        'Les références seront effacées.'
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
                (gx, dx, hy, by)
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
                st.warning(
                    'Présentez la feuille avec le code-barres '
                    'devant la caméra.'
                )
            else:
                st.warning(
                    'Aucun OF disponible.'
                )

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

            st.caption(
                'Chaque démarrage remet le compteur à zéro. '
                'Démarrez avant un cycle complet de production. '
                'Trois fermetures sont supposées correspondre à un câble.'
            )

            demarrage_desactive = (
                bool(erreur)
                or not all(refs.values())
                or not code_barres
            )

            if st.button(
                'Démarrer le comptage',
                disabled=demarrage_desactive
            ):
                try:
                    cam.demarrer(
                        refs,
                        marge,
                        seuil,
                        confirmation
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
            'Câbles estimés — groupes de 3',
            len(evenements) // 3
        )

        st.caption(
            f'{len(evenements) % 3} fermeture(s) '
            f'sans groupe complet.'
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
        # CSV AVEC OF
        # ====================================================

        if not actif and evenements:

            lignes_export = [
                {
                    'OF': of_compteur or code_barres or '',
                    **evenement
                }
                for evenement in evenements
            ]

            st.dataframe(
                lignes_export
            )

            fichier = io.StringIO()

            writer = csv.DictWriter(
                fichier,
                fieldnames=list(
                    lignes_export[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(
                lignes_export
            )

            nom_of = (
                of_compteur
                or code_barres
                or 'sans_of'
            )

            st.download_button(
                'Télécharger les fermetures CSV',
                fichier
                .getvalue()
                .encode('utf-8-sig'),
                f'fermetures_{nom_of}.csv',
                'text/csv'
            )

    panneau()
