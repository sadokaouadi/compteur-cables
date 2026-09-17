import cv2
import re
import zxingcpp


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 0

WIDTH = 1280
HEIGHT = 720

# Format STARZ attendu :
# exemple 48247-1
STARZ_PATTERN = re.compile(r"^\d{5}-\d$")


# =========================================================
# OUVERTURE CAMERA
# =========================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)


if not camera.isOpened():

    print("ERREUR : impossible d'ouvrir la camera.")

    raise SystemExit


print("=" * 55)
print(" SCANNER CODE-BARRES STARZ")
print("=" * 55)

print()
print("Camera ouverte.")
print("Placez le code-barres dans le rectangle.")
print()
print("Q = quitter")
print()


dernier_code = None


# =========================================================
# BOUCLE CAMERA
# =========================================================

while True:

    ok, frame = camera.read()


    if not ok:

        print(
            "ERREUR : impossible de lire une image."
        )

        break


    hauteur, largeur = frame.shape[:2]


    # =====================================================
    # ROI CENTRALE
    # =====================================================

    roi_width = int(
        largeur * 0.60
    )

    roi_height = int(
        hauteur * 0.30
    )


    x1 = int(
        (largeur - roi_width) / 2
    )

    y1 = int(
        (hauteur - roi_height) / 2
    )


    x2 = x1 + roi_width

    y2 = y1 + roi_height


    roi = frame[
        y1:y2,
        x1:x2
    ]


    # =====================================================
    # LECTURE ZXING-C++
    # =====================================================

    code_valide = None


    try:

        codes = zxingcpp.read_barcodes(
            roi
        )


        for code in codes:

            texte = code.text.strip()


            if STARZ_PATTERN.fullmatch(
                texte
            ):

                code_valide = texte

                break


    except Exception as erreur:

        print(
            "Erreur ZXing :",
            erreur
        )


    # =====================================================
    # AFFICHAGE ROI
    # =====================================================

    if code_valide:

        couleur = (
            0,
            255,
            0
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            couleur,
            3
        )


        cv2.putText(
            frame,

            "CODE DETECTE : "
            + code_valide,

            (
                x1,
                y1 - 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            couleur,

            2,

            cv2.LINE_AA
        )


        # Afficher dans le terminal
        # uniquement si nouveau code

        if (
            code_valide
            !=
            dernier_code
        ):

            print()
            print(
                "CODE DETECTE :",
                code_valide
            )

            print(
                "Validation STARZ : OK"
            )

            print()


            dernier_code = (
                code_valide
            )


    else:

        couleur = (
            0,
            255,
            255
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            couleur,
            2
        )


        cv2.putText(
            frame,

            "Placez le code-barres ici",

            (
                x1,
                y1 - 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            couleur,

            2,

            cv2.LINE_AA
        )


    # =====================================================
    # INFORMATIONS
    # =====================================================

    cv2.putText(
        frame,

        "Q : quitter",

        (
            20,
            hauteur - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    cv2.imshow(
        "Scanner STARZ - CODE 128",
        frame
    )


    # =====================================================
    # CLAVIER
    # =====================================================

    touche = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if touche == ord("q"):

        break


# =========================================================
# FIN
# =========================================================

camera.release()

cv2.destroyAllWindows()


print()
print("Scanner arrete.")