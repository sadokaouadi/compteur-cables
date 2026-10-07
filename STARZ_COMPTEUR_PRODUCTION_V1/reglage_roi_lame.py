
import time
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Réglage ROI lame",
    layout="wide",
)

MACHINES = {
    "Machine 1": {
        "camera_index": 0,
        "rotation": "90° gauche",
        "roi_compteur": (31, 43, 36, 48),
    },
    "Machine 2": {
        "camera_index": 0,
        "rotation": "90° droite",
        "roi_compteur": (29, 41, 45, 56),
    },
}


def orienter(image, orientation):
    if orientation == "90° gauche":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "90° droite":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def fermer_camera():
    cap = st.session_state.pop("roi_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


st.title("Réglage de la zone de déclenchement de la lame")
st.caption(
    "Outil temporaire : il sert seulement à trouver la petite ROI de la lame."
)

machine = st.selectbox("Machine", list(MACHINES.keys()))
config = MACHINES[machine]

if st.session_state.get("roi_machine_active") != machine:
    fermer_camera()
    st.session_state["roi_machine_active"] = machine
    st.session_state["roi_ref_gray"] = None
    st.session_state["roi_prev_gray"] = None
    st.session_state["roi_mvt_confirmations"] = 0
    st.session_state["roi_candidate_frames"] = 0
    st.session_state["roi_candidate_hits"] = 0
    st.session_state["roi_occlusion_cooldown"] = 0

gx1, gx2, gy1, gy2 = config["roi_compteur"]

st.info(
    f"{machine} — orientation {config['rotation']} — "
    f"ROI compteur actuelle H {gx1}–{gx2}% / V {gy1}–{gy2}%"
)

c1, c2 = st.columns(2)

with c1:
    h = st.slider(
        "Zone horizontale lame (%)",
        0,
        100,
        (gx1, gx2),
        key=f"h_{machine}",
    )

with c2:
    v = st.slider(
        "Zone verticale lame (%)",
        0,
        100,
        (gy1, gy2),
        key=f"v_{machine}",
    )

seuil_temporel = st.slider(
    "Seuil mouvement lame — différence entre 2 images successives",
    0.5,
    30.0,
    6.0,
    0.5,
)

seuil_occlusion = st.slider(
    "Seuil moyen main / occlusion — différence avec référence immobile",
    10.0,
    60.0,
    22.0,
    1.0,
)

seuil_ratio_occlusion = st.slider(
    "Pourcentage pixels modifiés = main / occlusion",
    30,
    95,
    65,
    5,
)

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Ouvrir / reconnecter caméra", use_container_width=True):
        fermer_camera()

        # Certains pilotes Windows lèvent une exception OpenCV quand on
        # force WIDTH / HEIGHT / FPS avec DirectShow. On ouvre d'abord la
        # caméra, puis on applique les réglages seulement si le pilote les
        # accepte. Sinon on garde automatiquement le format natif.
        cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(config["camera_index"])

        if not cap.isOpened():
            st.error(
                "Impossible d'ouvrir la caméra. Fermez l'application "
                "de production si elle utilise déjà cette caméra, puis réessayez."
            )
            st.stop()

        for propriete, valeur in (
            (cv2.CAP_PROP_FRAME_WIDTH, 1280),
            (cv2.CAP_PROP_FRAME_HEIGHT, 720),
            (cv2.CAP_PROP_FPS, 30),
        ):
            try:
                cap.set(propriete, valeur)
            except cv2.error:
                # Le pilote refuse ce réglage : on garde le format natif.
                pass

        # Vérifier qu'une vraie image peut être lue avant de continuer.
        ok_test, _ = cap.read()

        if not ok_test:
            cap.release()
            st.error(
                "Caméra ouverte mais aucune image reçue. "
                "Vérifiez qu'aucune autre application n'utilise la caméra."
            )
            st.stop()

        st.session_state["roi_cap"] = cap
        st.session_state["roi_ref_gray"] = None
        st.session_state["roi_prev_gray"] = None
        st.session_state["roi_mvt_confirmations"] = 0
        st.session_state["roi_candidate_frames"] = 0
        st.session_state["roi_candidate_hits"] = 0
        st.session_state["roi_occlusion_cooldown"] = 0
        st.rerun()

with col_b:
    if st.button("Prendre référence machine immobile", use_container_width=True):
        st.session_state["roi_ref_gray"] = "PENDING"

cap = st.session_state.get("roi_cap")

if cap is None or not cap.isOpened():
    st.warning("Cliquez sur « Ouvrir / reconnecter caméra ».")
    st.stop()


@st.fragment(run_every=0.15)
def afficher_live():
    cap = st.session_state.get("roi_cap")
    if cap is None or not cap.isOpened():
        st.error("Caméra non disponible.")
        return

    ok, frame = cap.read()
    if not ok:
        st.error("Lecture caméra impossible.")
        return

    frame = orienter(frame, config["rotation"])
    h_img, w_img = frame.shape[:2]

    x1 = int(w_img * h[0] / 100)
    x2 = int(w_img * h[1] / 100)
    y1 = int(h_img * v[0] / 100)
    y2 = int(h_img * v[1] / 100)

    x1 = max(0, min(x1, w_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(y1 + 1, min(y2, h_img))

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    ref = st.session_state.get("roi_ref_gray")
    prev = st.session_state.get("roi_prev_gray")

    # "ref" peut être soit la chaîne "PENDING", soit une image NumPy.
    if isinstance(ref, str) and ref == "PENDING":
        st.session_state["roi_ref_gray"] = gray.copy()
        ref = gray.copy()
        st.session_state["roi_prev_gray"] = gray.copy()
        prev = gray.copy()
        st.session_state["roi_mvt_confirmations"] = 0
        st.session_state["roi_candidate_frames"] = 0
        st.session_state["roi_candidate_hits"] = 0
        st.session_state["roi_occlusion_cooldown"] = 0

    diff_reference = None
    diff_temporelle = None
    pixels_modifies = None
    statut = "Référence non prise"

    if isinstance(ref, np.ndarray):
        if ref.shape != gray.shape:
            st.session_state["roi_ref_gray"] = None
            st.session_state["roi_prev_gray"] = None
            st.session_state["roi_mvt_confirmations"] = 0
            st.session_state["roi_candidate_frames"] = 0
            st.session_state["roi_candidate_hits"] = 0
            st.session_state["roi_occlusion_cooldown"] = 0
            statut = "ROI modifiée — reprendre la référence"
        else:
            # 1) Différence absolue avec l'état immobile :
            # sert surtout à reconnaître une main / grosse occlusion.
            diff_reference = float(
                cv2.absdiff(gray, ref).mean()
            )

            # 2) Différence entre images successives :
            # sert à détecter le DÉBUT DU MOUVEMENT, indépendamment
            # de la position ouverte/fermée de la lame.
            if isinstance(prev, np.ndarray) and prev.shape == gray.shape:
                diff_temporelle = float(
                    cv2.absdiff(gray, prev).mean()
                )
            else:
                diff_temporelle = 0.0

            # Pourcentage de pixels réellement différents de la référence.
            # Une main masque généralement une grande partie de la ROI,
            # contrairement au mouvement mécanique plus localisé.
            diff_ref_img = cv2.absdiff(gray, ref)
            pixels_modifies = float(
                np.mean(diff_ref_img >= 20) * 100.0
            )

            # Sauvegarder l'image actuelle pour le prochain passage.
            st.session_state["roi_prev_gray"] = gray.copy()

            occlusion = (
                diff_reference >= seuil_occlusion
                or pixels_modifies >= float(seuil_ratio_occlusion)
            )

            candidate_frames = int(
                st.session_state.get("roi_candidate_frames", 0)
            )
            candidate_hits = int(
                st.session_state.get("roi_candidate_hits", 0)
            )
            cooldown = int(
                st.session_state.get("roi_occlusion_cooldown", 0)
            )

            if occlusion:
                # Main / grosse occlusion :
                # verrouiller le démarrage pendant 15 images (~0,5 s à 30 FPS)
                # APRÈS disparition de la main.
                statut = "OCCLUSION / MAIN — DÉCLENCHEMENT REFUSÉ"
                st.session_state["roi_candidate_frames"] = 0
                st.session_state["roi_candidate_hits"] = 0
                st.session_state["roi_mvt_confirmations"] = 0
                st.session_state["roi_occlusion_cooldown"] = 15

            elif cooldown > 0:
                # Très important : la sortie rapide de la main crée souvent
                # une énorme différence entre deux images. On l'ignore ici.
                cooldown -= 1
                st.session_state["roi_occlusion_cooldown"] = cooldown
                st.session_state["roi_candidate_frames"] = 0
                st.session_state["roi_candidate_hits"] = 0

                # Réinitialiser l'image précédente pour que la sortie de main
                # ne puisse pas devenir un faux mouvement mécanique.
                st.session_state["roi_prev_gray"] = gray.copy()

                statut = (
                    "APRÈS OCCLUSION — ATTENTE SÉCURITÉ "
                    f"({cooldown}/15)"
                )

            else:
                mouvement = diff_temporelle >= seuil_temporel

                # Confirmation plus stricte :
                # 6 images observées, au moins 3 vraies images de mouvement.
                if candidate_frames == 0:
                    if mouvement:
                        st.session_state["roi_candidate_frames"] = 1
                        st.session_state["roi_candidate_hits"] = 1
                        statut = "MOUVEMENT POSSIBLE — vérification"
                    else:
                        statut = "IMMOBILE / ATTENTE"
                else:
                    candidate_frames += 1
                    if mouvement:
                        candidate_hits += 1

                    st.session_state["roi_candidate_frames"] = candidate_frames
                    st.session_state["roi_candidate_hits"] = candidate_hits

                    if candidate_frames >= 6:
                        if candidate_hits >= 3:
                            statut = "MOUVEMENT LAME CONFIRMÉ"
                        else:
                            statut = "IMMOBILE / ATTENTE"

                        st.session_state["roi_candidate_frames"] = 0
                        st.session_state["roi_candidate_hits"] = 0
                    else:
                        statut = (
                            "MOUVEMENT POSSIBLE — vérification "
                            f"({candidate_frames}/6)"
                        )

    preview = frame.copy()
    couleur = (0, 255, 0)
    if statut == "MOUVEMENT LAME CONFIRMÉ":
        couleur = (0, 0, 255)
    elif statut == "OCCLUSION / MAIN — DÉCLENCHEMENT REFUSÉ":
        couleur = (0, 165, 255)

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2 - 1, y2 - 1),
        couleur,
        3,
    )

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    a, b = st.columns([2, 1])

    with a:
        st.image(
            preview_rgb,
            caption="Vue caméra — rectangle = ROI lame",
            use_container_width=True,
        )

    with b:
        st.image(
            roi_rgb,
            caption="ROI lame",
            use_container_width=True,
        )

        if statut == "MOUVEMENT LAME CONFIRMÉ":
            st.error(statut)
        elif statut == "OCCLUSION / MAIN — DÉCLENCHEMENT REFUSÉ":
            st.warning(statut)
        elif statut == "IMMOBILE / ATTENTE":
            st.success(statut)
        elif statut.startswith("APRÈS OCCLUSION"):
            st.warning(statut)
        else:
            st.info(statut)

        if diff_temporelle is not None:
            st.metric(
                "Mouvement instantané lame",
                f"{diff_temporelle:.2f}"
            )

        if diff_reference is not None:
            st.metric(
                "Différence avec référence immobile",
                f"{diff_reference:.2f}"
            )

        if pixels_modifies is not None:
            st.metric(
                "Pixels modifiés dans la ROI",
                f"{pixels_modifies:.1f} %"
            )

    st.code(
        f"{machine}\n"
        f"Zone horizontale : {h[0]} → {h[1]} %\n"
        f"Zone verticale   : {v[0]} → {v[1]} %"
    )


afficher_live()

st.divider()
st.markdown(
    """
### Comment tester — V6 anti-main rapide
1. Gardez la même ROI.
2. Machine immobile : prenez la référence.
3. Passez la main vite : **OCCLUSION / MAIN**, puis **APRÈS OCCLUSION — ATTENTE SÉCURITÉ**.
4. La sortie de la main ne doit jamais devenir **MOUVEMENT LAME CONFIRMÉ**.
5. Lancez réellement la machine quand la zone est claire : le mouvement doit être confirmé sur plusieurs images.
6. En production, un buffer conservera les images précédentes, donc cette confirmation retardée ne fera pas perdre le début des 800 images.
"""
)
