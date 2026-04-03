import os
import sys
import hashlib
from ultralytics import YOLO
from keystoneauth1.identity import v3
from keystoneauth1 import session
from glanceclient import Client as GlanceClient
from swiftclient import client as swift_client

# --- CONFIGURAZIONE OPENSTACK ---
AUTH_URL = "http://192.168.188.131/identity/v3" 
USERNAME = "admin"
PASSWORD = "secret" 
PROJECT_NAME = "admin"
USER_DOMAIN_NAME = "Default"
PROJECT_DOMAIN_NAME = "Default"

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

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 cli_ia.py /percorso/immagine.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Autenticazione
        auth = v3.Password(auth_url=AUTH_URL, username=USERNAME, password=PASSWORD,
                           project_name=PROJECT_NAME, user_domain_name=USER_DOMAIN_NAME,
                           project_domain_name=PROJECT_DOMAIN_NAME)
        sess = session.Session(auth=auth)
        glance = GlanceClient('2', session=sess)
        swift = swift_client.Connection(session=sess)

        # Sincronizzazione modello (Glance)
        remote_image = next(glance.images.list(filters={'name': MODEL_NAME_ON_GLANCE}))
        if get_local_md5(LOCAL_FILE_PATH) != remote_image.checksum:
            print(f"Aggiornamento modello da Glance...")
            with open(LOCAL_FILE_PATH, 'wb') as f:
                for chunk in glance.images.data(remote_image.id):
                    f.write(chunk)

        # Inferenza e visualizzazione terminale
        print(f"\nAnalisi AI in corso su: {os.path.basename(image_path)}")
        model = YOLO(LOCAL_FILE_PATH)
        results = model(image_path, verbose=False)

        print("-" * 30)
        print(f"{'OGGETTO':<15} | {'CONFIDENZA':<10}")
        print("-" * 30)

        # Ciclo per stampare i risultati nel terminale
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"{label:<15} | {confidence:.2%}")

        print("-" * 30)

        # Salvataggio temporaneo e Upload su Swift
        res_filename = f"terminale_{os.path.basename(image_path)}"
        results[0].save(res_filename) 

        container_name = "immagini_ia"
        swift.put_container(container_name)
        
        with open(res_filename, 'rb') as f:
            swift.put_object(container_name, res_filename, contents=f)
        
        print(f"Risultato caricato su Swift nel container '{container_name}'")

        # Pulizia locale
        if os.path.exists(res_filename):
            os.remove(res_filename)

    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()
