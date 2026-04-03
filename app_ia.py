import streamlit as st
import os
import hashlib
import io
from PIL import Image
from ultralytics import YOLO
from keystoneauth1.identity import v3
from keystoneauth1 import session
from glanceclient import Client as GlanceClient
from swiftclient import client as swift_client

# --- CONFIGURAZIONE ---
AUTH_URL = "http://192.168.188.131/identity/v3"
USERNAME = "admin"
PASSWORD = "secret"
PROJECT_NAME = "admin"

# --- CONFIGURAZIONE MODELLO ---
MODEL_NAME_ON_GLANCE = "YOLO11-Model-v1" 
LOCAL_FILE_PATH = "YOLO11-Model-v1.pt" 

def get_local_md5(filename):
    if not os.path.exists(filename):
        return None
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# --- SIDEBAR (Grafica Tech Stack) ---
with st.sidebar:
    st.markdown("### Tech Stack 2026")
    st.markdown("- **Model:** YOLO11 Nano")
    st.markdown("- **Infrastructure:** OpenStack")
    st.markdown("- **Library:** Ultralytics")

st.title("AI Cloud Analyzer")

try:
    auth = v3.Password(auth_url=AUTH_URL, username=USERNAME, password=PASSWORD,
                       project_name=PROJECT_NAME, user_domain_name="Default",
                       project_domain_name="Default")
    sess = session.Session(auth=auth)
    glance = GlanceClient('2', session=sess)
    swift = swift_client.Connection(session=sess)

    # Verifica e sincronizzazione modello
    remote_img = next(glance.images.list(filters={'name': MODEL_NAME_ON_GLANCE}))
    remote_checksum = remote_img.checksum
    local_checksum = get_local_md5(LOCAL_FILE_PATH)

    if local_checksum != remote_checksum:
        with st.status(f"Scaricamento {MODEL_NAME_ON_GLANCE}...", expanded=True) as status:
            with open(LOCAL_FILE_PATH, 'wb') as f:
                for chunk in glance.images.data(remote_img.id):
                    f.write(chunk)
            status.update(label="Modello scaricato!", state="complete", expanded=False)
    else:
        st.info(f"Modello '{MODEL_NAME_ON_GLANCE}' pronto.")

    # Caricamento immagini
    uploaded_files = st.file_uploader("Carica le tue immagini...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        model = YOLO(LOCAL_FILE_PATH, task="detect")
        
        for uploaded_file in uploaded_files:
            with st.expander(f"Dettagli Analisi: {uploaded_file.name}", expanded=True):
                # Analisi
                input_image = Image.open(uploaded_file)
                results = model(input_image, verbose=False)
                
                # Rendering immagine
                res_plotted = results[0].plot()
                res_image = Image.fromarray(res_plotted[:, :, ::-1])
                
                # Layout a due colonne (Immagine | Testo)
                col_img, col_txt = st.columns([2, 1])
                
                with col_img:
                    st.image(res_image, caption="Rilevamento YOLO11", use_container_width=True)

                with col_txt:
                    # Banner verde di stato salvataggio
                    st.success("Dato salvato su Swift")
                    
                    st.markdown("**Oggetti identificati:**")
                    # Contiamo gli oggetti per la lista (es. DOG: 1)
                    counts = {}
                    for box in results[0].boxes:
                        label = model.names[int(box.cls[0])].upper()
                        counts[label] = counts.get(label, 0) + 1
                    
                    for label, count in counts.items():
                        st.write(f"- {label}: {count}")

                # Upload su Swift (Automatico)
                nome_cloud = f"UI_{uploaded_file.name}"
                img_buffer = io.BytesIO()
                res_image.save(img_buffer, format="JPEG")
                img_buffer.seek(0)
                
                swift.put_container("immagini_ia")
                swift.put_object("immagini_ia", nome_cloud, contents=img_buffer)
                st.success(f"Archiviato su Swift nel container immagini_ia come: {nome_cloud}")
                st.markdown("---")


except Exception as e:
    st.error(f"Errore: {e}")
