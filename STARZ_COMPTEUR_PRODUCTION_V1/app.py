import streamlit as st

from camera_direct import afficher_camera_production


st.set_page_config(
    page_title="STARZ — Compteur de câbles",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

afficher_camera_production()
