import streamlit as st
import os
import hashlib
import io
import uuid
from PIL import Image
from ultralytics import YOLO
from keystoneauth1.identity import v3
from keystoneauth1 import session
from glanceclient import Client as GlanceClient
from swiftclient import client as swift_client

# --- CONFIGURAZIONE SICURA ---
AUTH_URL = os.environ.get("OS_AUTH_URL")
USERNAME = os.environ.get("OS_USERNAME")
PASSWORD = os.environ.get("OS_PASSWORD")
PROJECT_NAME = os.environ.get("OS_PROJECT_NAME")
USER_DOMAIN = os.environ.get("OS_USER_DOMAIN_NAME", "Default")
PROJECT_DOMAIN = os.environ.get("OS_PROJECT_DOMAIN_NAME", "Default")

def get_local_md5(filename):
    if not os.path.exists(filename):
        return None
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Tech Stack 2026")
    st.info(f"Utente corrente: {USERNAME}")
    
    st.markdown("---")
    st.markdown("### Selezione Modello AI")
    modelli_disponibili = {
        "YOLO11 Nano (Veloce)": "YOLO11-Model-v1",
        "YOLO11 Small (Bilanciato)": "YOLO11-Small",
        "YOLO11 Segmentation (Sagome)": "YOLO11-Segmentation",
        "YOLO11 Classifier (Etichette)": "YOLO11-Classifier"
    }
    scelta_utente = st.selectbox("Scegli il modello da Glance:", list(modelli_disponibili.keys()))
    MODEL_NAME_ON_GLANCE = modelli_disponibili[scelta_utente]
    LOCAL_FILE_PATH = f"{MODEL_NAME_ON_GLANCE}.pt"

st.title("AI Cloud Analyzer")

if not all([AUTH_URL, USERNAME, PASSWORD, PROJECT_NAME]):
    st.error("Errore di configurazione: Variabili d'ambiente non trovate.")
    st.stop()

try:
    auth = v3.Password(
        auth_url=AUTH_URL, 
        username=USERNAME, 
        password=PASSWORD,
        project_name=PROJECT_NAME, 
        user_domain_name=USER_DOMAIN,
        project_domain_name=PROJECT_DOMAIN
    )
    sess = session.Session(auth=auth)
    glance = GlanceClient('2', session=sess)
    swift = swift_client.Connection(session=sess)

    remote_img = next(glance.images.list(filters={'name': MODEL_NAME_ON_GLANCE}))
    remote_checksum = remote_img.checksum
    local_checksum = get_local_md5(LOCAL_FILE_PATH)

    if local_checksum != remote_checksum:
        with st.status(f"Sincronizzazione {MODEL_NAME_ON_GLANCE}...", expanded=True) as status:
            with open(LOCAL_FILE_PATH, 'wb') as f:
                for chunk in glance.images.data(remote_img.id):
                    f.write(chunk)
            status.update(label="Modello pronto!", state="complete", expanded=False)
    else:
        st.info(f"Modello '{scelta_utente}' in uso.")

    uploaded_files = st.file_uploader("Carica le immagini...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        model = YOLO(LOCAL_FILE_PATH)
        
        for uploaded_file in uploaded_files:
            with st.expander(f"Analisi: {uploaded_file.name}"):
                input_image = Image.open(uploaded_file)
                results = model(input_image, verbose=False)
                
                res_plotted = results[0].plot()
                res_image = Image.fromarray(res_plotted[:, :, ::-1])
                
                col_img, col_txt = st.columns([2, 1])
                
                with col_img:
                    st.image(res_image, use_container_width=True)

                with col_txt:
                    st.markdown("**Risultati:**")
                    if hasattr(results[0], 'probs') and results[0].probs is not None:
                        top1_idx = results[0].probs.top1
                        label = results[0].names[top1_idx].upper()
                        conf = results[0].probs.top1conf.item() * 100
                        st.write(f"Classe: {label}")
                        st.write(f"Confidenza: {conf:.1f}%")
                    else:
                        counts = {}
                        for box in results[0].boxes:
                            label = model.names[int(box.cls[0])].upper()
                            counts[label] = counts.get(label, 0) + 1
                        for label, count in counts.items():
                            st.write(f"- {label}: {count}")

                # --- GENERAZIONE NOME CON UUID E LABEL UI_ ---
                unique_id = uuid.uuid4().hex[:8]
                nome_cloud = f"UI_{unique_id}_{uploaded_file.name}"
                
                img_buffer = io.BytesIO()
                res_image.save(img_buffer, format="JPEG")
                img_buffer.seek(0)
                
                container = "immagini_ia"
                swift.put_container(container)
                swift.put_object(container, nome_cloud, contents=img_buffer)
                
                st.success(f"Archiviato univocamente come: {nome_cloud}")

except Exception as e:
    st.error(f"Errore critico: {e}")
