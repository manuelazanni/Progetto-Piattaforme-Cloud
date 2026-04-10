import os
import sys
import hashlib
import uuid
from ultralytics import YOLO
from keystoneauth1.identity import v3
from keystoneauth1 import session
from glanceclient import Client as GlanceClient
from swiftclient import client as swift_client

# --- CONFIGURAZIONE OPENSTACK ---
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

def main():
    # Controllo argomenti
    if len(sys.argv) < 2:
        print("Uso: python3 cli_ia.py /percorso/immagine.jpg [NOME_MODELLO_GLANCE]")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # Se l'utente passa un secondo argomento, usa quello, altrimenti usa il Nano
    model_name_glance = sys.argv[2] if len(sys.argv) > 2 else "YOLO11-Model-v1"
    local_file_path = f"{model_name_glance}.pt"

    if not all([AUTH_URL, USERNAME, PASSWORD, PROJECT_NAME]):
        print("Errore: Variabili d'ambiente non trovate. Esegui 'source ai_user.env'")
        sys.exit(1)

    try:
        # Autenticazione
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

        # Sincronizzazione modello scelto da Glance
        print(f"Verifica modello: {model_name_glance}...")
        remote_image = next(glance.images.list(filters={'name': model_name_glance}))
        
        if get_local_md5(local_file_path) != remote_image.checksum:
            print(f"Download modello {model_name_glance} in corso...")
            with open(local_file_path, 'wb') as f:
                for chunk in glance.images.data(remote_image.id):
                    f.write(chunk)

        # Inferenza
        print(f"\nAnalisi AI (Terminale) con {model_name_glance} su: {os.path.basename(image_path)}")
        model = YOLO(local_file_path)
        results = model(image_path, verbose=False)

        print("-" * 40)
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            top1_idx = results[0].probs.top1
            label = results[0].names[top1_idx].upper()
            conf = results[0].probs.top1conf.item()
            print(f"CLASSE RILEVATA: {label} ({conf:.2%})")
        else:
            print(f"{'OGGETTO':<20} | {'CONFIDENZA':<10}")
            print("-" * 40)
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])].upper()
                confidence = float(box.conf[0])
                print(f"{label:<20} | {confidence:.2%}")
        print("-" * 40)

        # Generazione nome con UUID e label terminale_
        unique_id = uuid.uuid4().hex[:8]
        base_name = os.path.basename(image_path)
        nome_cloud = f"terminale_{unique_id}_{base_name}"
        
        temp_local_file = f"temp_{nome_cloud}"
        results[0].save(temp_local_file) 

        # Upload su Swift
        container_name = "immagini_ia"
        swift.put_container(container_name)
        
        with open(temp_local_file, 'rb') as f:
            swift.put_object(container_name, nome_cloud, contents=f)
        
        print(f"Risultato caricato su Swift come: {nome_cloud}")

        if os.path.exists(temp_local_file):
            os.remove(temp_local_file)

    except StopIteration:
        print(f"Errore: Il modello '{model_name_glance}' non esiste su Glance.")
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()
