"""Caméra locale : acquisition et analyse continues, affichage périodique."""
import csv
import io
import threading
import time
from collections import deque
import platform

import cv2
import streamlit as st
from analyse import preparer_references, estimer_etat


class Compteur:
    def __init__(self, confirmation=1):
        self.confirmation = confirmation
        self.stable = self.candidat = self.debut = None
        self.repetitions = 0
        self.evenements = []
        self.initiale = False

    def ajouter(self, etat, numero, secondes):
        if etat == 'INDETERMINE':
            self.candidat = None
            self.repetitions = 0
            return
        if etat != self.candidat:
            self.candidat, self.repetitions = etat, 0
            self.premier = (numero, secondes)
        self.repetitions += 1
        if self.repetitions < self.confirmation or etat == self.stable:
            return
        precedent, self.stable = self.stable, etat
        if etat == 'FERME':
            self.debut = self.premier if precedent == 'OUVERT' else None
            self.initiale |= precedent is None
        elif precedent == 'FERME' and self.debut is not None:
            self.evenements.append({'Fermeture': len(self.evenements)+1,
                'Image_debut': self.debut[0], 'Secondes_debut': round(self.debut[1], 3),
                'Image_reouverture': self.premier[0],
                'Secondes_reouverture': round(self.premier[1], 3),
                'Image_confirmation': numero})
            self.debut = None


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
        self.heartbeat = time.monotonic()
        self.depart = self.heartbeat
        self.thread = threading.Thread(target=self._lire, args=(index,), daemon=True)
        self.thread.start()

    def _lire(self, index):
        cap = None
        try:
            backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                raise ValueError('Caméra inaccessible : fermez Caméra Windows, puis essayez un autre index.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            cap.set(cv2.CAP_PROP_FPS, 30)
            while not self.stop_event.is_set():
                if time.monotonic()-self.heartbeat > 30:
                    raise ValueError('Connexion à la page interrompue : caméra arrêtée.')
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Lecture caméra interrompue. Le comptage est partiel.')
                now = time.monotonic()
                h, w = image.shape[:2]
                g,d,t,b = self.limites
                x1,x2,y1,y2 = int(w*g/100),int(w*d/100),int(h*t/100),int(h*b/100)
                zone = cv2.cvtColor(image[y1:y2,x1:x2], cv2.COLOR_BGR2RGB)
                if not zone.size:
                    raise ValueError('Zone vide : modifiez les limites.')
                with self.lock:
                    self.numero += 1
                    self.derniere = (image, zone, (x1,y1,x2,y2))
                    self.historique.append((self.numero, zone.copy()))
                    if self.actif:
                        gris = cv2.cvtColor(zone, cv2.COLOR_RGB2GRAY)
                        self.etat, _, _ = estimer_etat(gris, self.refs, self.marge, self.seuil)
                        self.compteur.ajouter(self.etat, self.numero, now-self.debut_test)
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
                raise ValueError('Reconnectez la caméra avant de démarrer.')
            self.refs = preparer_references(refs)
            if any((a == b).all() for a in self.refs['ouvert'] for b in self.refs['ferme']):
                raise ValueError('Les références ouverte et fermée sont identiques.')
            self.compteur = Compteur(confirmation)
            self.marge, self.seuil = marge, seuil
            self.debut_test = time.monotonic()
            self.actif = True

    def arreter(self):
        with self.lock:
            self.actif = False

    def fermer(self):
        self.arreter()
        self.stop_event.set()
        self.thread.join(timeout=2)


def afficher_camera():
    if not hasattr(st, 'fragment'):
        st.error('Mettez Streamlit à jour : python -m pip install --upgrade streamlit')
        return
    st.info('Caméra du PC qui exécute Streamlit. Fermez les autres applications utilisant la caméra.')
    cam = st.session_state.get('camera_direct')
    connectee = cam is not None
    index = int(st.number_input('Index caméra (0, puis 1 ou 2 si nécessaire)', 0, 10, 1, disabled=connectee))
    gx, dx = st.slider('Zone horizontale (%)', 0, 100, (44,54), disabled=connectee)
    hy, by = st.slider('Zone verticale (%)', 0, 100, (53,65), disabled=connectee)
    st.caption('Pour modifier la zone ou changer de caméra, déconnectez puis reconnectez. Les références seront effacées.')
    if st.button('Connecter la caméra', disabled=connectee):
        if gx >= dx or hy >= by:
            st.error('Choisissez une zone non vide.')
        else:
            st.session_state.camera_direct = Camera(index, (gx,dx,hy,by))
            st.session_state.cam_refs = {'ouvert': [], 'ferme': []}
            st.session_state.pop('cam_gel', None)
            st.rerun()
    if cam is None:
        return
    if st.button('Déconnecter la caméra'):
        cam.fermer()
        del st.session_state.camera_direct
        st.rerun()

    @st.fragment(run_every=0.2)
    def panneau():
        cam.heartbeat = time.monotonic()
        with cam.lock:
            actif = cam.actif
            dernier = cam.derniere
            erreur = cam.erreur
            evenements = list(cam.compteur.evenements)
            incomplet = cam.compteur.debut is not None
            initiale = cam.compteur.initiale
            etat = cam.etat
            numero = cam.numero
        if erreur:
            st.error(erreur)
        if dernier is None:
            st.info('Ouverture de la caméra…')
            return
        image, zone, (x1,y1,x2,y2) = dernier
        apercu = image.copy()
        cv2.rectangle(apercu, (x1,y1), (x2-1,y2-1), (0,255,0), 2)
        a,b = st.columns(2)
        a.image(apercu, channels='BGR', width=440)
        b.image(zone, width=260, caption='Zone analysée')
        st.caption(f'{numero} images reçues. Cadence moyenne reçue : {numero/max(0.001,time.monotonic()-cam.depart):.1f} images/s. Affichage ralenti, analyse de chaque image reçue.')
        refs = st.session_state.cam_refs
        if not actif:
            st.write('Références : filmez quelques mouvements, puis figez les images récentes pour sélectionner ouvert et fermé.')
            if st.button('Figer les images récentes', disabled=bool(erreur)):
                with cam.lock:
                    st.session_state.cam_gel = list(cam.historique)
                st.session_state.pop('cam_selection', None)
            gel = st.session_state.get('cam_gel', [])
            if gel:
                selection = st.select_slider('Image à classer', options=list(range(len(gel))), key='cam_selection')
                n, z = gel[selection]
                st.image(z, width=280, caption=f'Image {n} — sélection figée')
                for nom, libelle in [('ouvert','OUVERT'), ('ferme','FERMÉ')]:
                    if st.button(f'Ajouter comme {libelle}'):
                        if any(r['numero']==n for valeurs in refs.values() for r in valeurs):
                            st.warning('Cette image est déjà classée. Effacez les références pour corriger.')
                        else:
                            refs[nom].append({'numero':n, 'image':z.copy()})
            st.caption(f"Ouvert : {len(refs['ouvert'])} référence(s) ; fermé : {len(refs['ferme'])} référence(s).")
            if st.button('Effacer les références'):
                refs['ouvert'].clear()
                refs['ferme'].clear()
            marge = st.number_input('Marge minimale', 0.0, 20.0, 0.5, 0.1)
            seuil = st.number_input('Différence maximale', 1.0, 100.0, 40.0, 1.0)
            confirmation = st.number_input('Images de confirmation', 1, 10, 1)
            st.caption('Chaque démarrage remet le compteur à zéro. Démarrez avant un cycle complet de production. Trois fermetures sont supposées correspondre à un câble.')
            if st.button('Démarrer le comptage', disabled=bool(erreur) or not all(refs.values())):
                try:
                    cam.demarrer(refs, marge, seuil, confirmation)
                    st.rerun(scope='fragment')
                except ValueError as exc:
                    st.error(str(exc))
        else:
            if st.button('Arrêter le comptage'):
                cam.arreter()
                st.rerun(scope='fragment')
        st.write(f"Comptage : {'EN COURS' if actif else 'ARRÊTÉ'} — état estimé : {etat}")
        st.metric('Fermetures complètes', len(evenements))
        st.metric('Câbles estimés — groupes de 3', len(evenements)//3)
        st.caption(f'{len(evenements)%3} fermeture(s) sans groupe complet.')
        if initiale:
            st.warning('Début observé fermé : la première fermeture sans ouverture préalable est exclue.')
        if incomplet and not actif:
            st.warning('Une fermeture était encore en cours à l’arrêt.')
        if not actif and evenements:
            st.dataframe(evenements)
            fichier = io.StringIO()
            writer = csv.DictWriter(fichier, fieldnames=list(evenements[0]))
            writer.writeheader()
            writer.writerows(evenements)
            st.download_button('Télécharger les fermetures CSV', fichier.getvalue().encode('utf-8-sig'), 'fermetures_camera.csv', 'text/csv')
    panneau()
