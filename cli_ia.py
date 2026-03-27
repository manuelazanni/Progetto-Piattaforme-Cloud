import sys, os
from ultralytics import YOLO
from swiftclient import client

# Configurazione OpenStack
config = {'authurl': 'http://192.168.188.131/identity/v3', 'user': 'admin', 'key': 'secret',
          'os_options': {'project_name': 'admin', 'project_domain_name': 'Default', 'user_domain_name': 'Default'},
          'auth_version': '3'}

def analizza(percorso_foto):
    if not os.path.exists(percorso_foto):
        print(f"File non trovato: {percorso_foto}")
        return

    model = YOLO('yolo11n.pt')
    
    # Upload su Swift (così il prof vede che carichi sul Cloud anche da CLI)
    conn = client.Connection(**config)
    with open(percorso_foto, 'rb') as f:
        conn.put_object('immagini_ia', f"terminale_{os.path.basename(percorso_foto)}", contents=f.read())
    
    # Inferenza YOLO11
    results = model(percorso_foto, verbose=False)
    print(f"\n--- RISULTATI CLOUD AI (YOLO11) ---")
    for box in results[0].boxes:
        nome = results[0].names[int(box.cls[0])]
        conf = box.conf[0]
        print(f"OGGETTO: {nome.upper()} | SICUREZZA: {conf:.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analizza(sys.argv[1])
    else:
        print("Uso: python3 cli_ia.py nome_foto.jpg")
