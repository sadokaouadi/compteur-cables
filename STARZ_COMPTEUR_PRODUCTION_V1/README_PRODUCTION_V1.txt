STARZ — COMPTEUR DE CÂBLES — PRODUCTION V1
==========================================

OBJECTIF
--------
Cette version est destinée à l'opérateur sur le PC de la machine.

Flux opérateur :
1. Scanner le Work Order (facultatif) ou continuer sans WO.
2. Saisir la longueur du câble.
3. Saisir la quantité à produire.
4. Valider la commande.
5. Cliquer sur "Démarrer la production".
6. Continuer à utiliser la machine normalement.
7. L'application initialise automatiquement le compteur sur 800 images,
   crée 2 références OUVERT + 2 références FERME, récupère les câbles
   déjà coupés puis poursuit le compteur global.
8. Cliquer sur "Terminer la commande" à la fin.

PARAMÈTRES TECHNIQUES MASQUÉS
-----------------------------
Caméra : index 0
Rotation : 90° gauche
ROI : horizontal 31–43 %, vertical 36–48 %
Exposition : verrouillée
Calibration automatique : 800 images
Références : 2 OUVERT + 2 FERME
Confirmation : 1

SAUVEGARDE AUTOMATIQUE
----------------------
Les résultats sont enregistrés automatiquement dans :

historique_production\AAAA-MM-JJ\

Chaque commande produit :
- *_BILAN.csv
- *_EVENEMENTS.csv

L'enregistrement est mis à jour pendant la production et finalisé lorsque
l'opérateur clique sur "Terminer la commande".

LANCEMENT
---------
Depuis PowerShell, dans le dossier du projet :

.\venv\Scripts\python.exe -m streamlit run app.py

IMPORTANT
---------
Conserver la version de développement actuelle en sauvegarde.
Valider cette interface Production V1 sur la machine avant un déploiement
permanent.
