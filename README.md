# Progetto Piattaforme Cloud: Architettura cloud per object detection

Questo progetto implementa un sistema di intelligenza artificiale per il riconoscimento di oggetti nelle immagini integrato all'interno di un'infrastruttura cloud **OpenStack**.


## Obiettivi del progetto
Il progetto illustra lo sviluppo e l'integrazione di un AI service plugin all'interno di un'architettura cloud basata su OpenStack. L'obiettivo principale è estendere le funzionalità core della piattaforma, introducendo un modulo di analisi intelligente delle immagini basato sull'algoritmo YOLO11.

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
