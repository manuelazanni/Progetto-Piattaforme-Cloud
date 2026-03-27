import streamlit as st
from ultralytics import YOLO
from swiftclient import client
import PIL.Image
import io
import numpy as np

# --- Configurazione OpenStack (Verifica che l'IP sia corretto) ---
config = {
    'authurl': 'http://192.168.188.131/identity/v3',
    'user': 'admin', 
    'key': 'secret',
    'os_options': {'project_name': 'admin', 'project_domain_name': 'Default', 'user_domain_name': 'Default'},
    'auth_version': '3'
}

st.set_page_config(page_title="Manuela State-of-the-Art AI", page_icon="🎯")

@st.cache_resource
def load_latest_yolo():
    # Carichiamo YOLO11 Nano: l'ultima frontiera della Computer Vision (2026)
    return YOLO('yolo11n.pt') 

model = load_latest_yolo()

st.title("Progetto Piattafrome Cloud Computing")
st.write("Analisi multi-oggetto ad alta precisione con integrazione **OpenStack Object Storage**.")

uploaded_files = st.file_uploader("Carica le tue immagini...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if st.button('Esegui Inferenza nel Cloud'):
        try:
            conn = client.Connection(**config)
            
            for uploaded_file in uploaded_files:
                img_bytes = uploaded_file.getvalue()
                
                # 1. Archiviazione su OpenStack Swift
                nome_remoto = f"yolo11_{uploaded_file.name}"
                conn.put_object('immagini_ia', nome_remoto, contents=img_bytes)
                
                # 2. Analisi IA con YOLO11
                img = PIL.Image.open(io.BytesIO(img_bytes))
                results = model.predict(source=img, conf=0.25)
                
                with st.expander(f"Dettagli Analisi: {uploaded_file.name}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Visualizzazione con Bounding Boxes (Riquadri colorati)
                        res_plotted = results[0].plot()
                        st.image(res_plotted, channels="BGR", caption="Rilevamento YOLO11", use_container_width=True)
                        
                    with col2:
                        st.success(f"Dato salvato su Swift")
                        st.write("**Oggetti identificati:**")
                        
                        labels = results[0].boxes.cls.tolist()
                        names = results[0].names
                        unique_labels = set(labels)
                        
                        if unique_labels:
                            for label_id in unique_labels:
                                name = names[int(label_id)]
                                count = labels.count(label_id)
                                st.write(f"- {name.upper()}: `{count}`")
                        else:
                            st.info("Nessun oggetto rilevato.")

            st.balloons()
        except Exception as e:
            st.error(f"Errore di sistema: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Tech Stack 2026")
st.sidebar.write("- Model: **YOLO11 Nano**")
st.sidebar.write("- Infrastructure: **OpenStack**")
st.sidebar.write("- Library: **Ultralytics**")
