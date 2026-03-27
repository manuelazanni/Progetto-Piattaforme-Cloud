# Progetto Piattaforme Cloud: Architettura cloud per object detection distribuita

Questo progetto implementa un sistema di intelligenza artificiale per il riconoscimento di oggetti nelle immagini integrato all'interno di un'infrastruttura cloud **OpenStack**.


## Obiettivi del progetto
L'obiettivo è dimostrare l'integrazione tra algoritmi di Computer Vision (YOLO11) e i servizi core di un ambiente IaaS, gestendo il ciclo di vita dei dati e dei modelli interamente sul cloud.

## Architettura e servizi OpenStack utilizzati

Il sistema non lavora in locale, ma sfrutta i seguenti componenti di OpenStack:

* **Keystone (Identity Service):** gestisce l'autenticazione tramite API v3 e la separazione dei progetti.
* **Glance (Image Service):** utilizzato come repository centralizzato per i pesi del modello (`yolo11n.pt`). Il modello è registrato come artefatto binario.
* **Swift (Object Storage):** utilizzato per l'archiviazione persistente delle immagini analizzate. Ogni upload segue una naming convention specifica (`terminale_` o `yolo11_`) per distinguere la sorgente.
* **Nova (Compute):** fornisce le risorse di calcolo necessarie per l'esecuzione degli script Python e del motore di inferenza.


## Struttura del repository
* `app_ia.py`: interfaccia web interattiva realizzata con Streamlit.
* `cli_ia.py`: script per l'esecuzione da riga di comando.
* `.gitignore`: configurazione per escludere il modello pesant e file temporanei, mantenendo il repository leggero.
