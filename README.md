# Progetto Piattaforme Cloud: Architettura cloud per object detection

Questo progetto implementa un sistema di intelligenza artificiale per il riconoscimento di oggetti nelle immagini integrato all'interno di un'infrastruttura cloud **OpenStack**.


## Obiettivi del progetto
Il progetto illustra lo sviluppo e l'integrazione di un AI service plugin all'interno di un'architettura cloud basata su OpenStack. L'obiettivo principale è estendere le funzionalità core della piattaforma, introducendo un modulo di analisi intelligente delle immagini basato sull'algoritmo YOLO11.

## Architettura e servizi OpenStack utilizzati

Il sistema non lavora in locale, ma sfrutta i seguenti componenti di OpenStack:

* **Keystone (Identity Service):** gestisce l'autenticazione tramite API v3 e la separazione dei progetti.
* **Glance (Image Service):** utilizzato come repository centralizzato per i pesi dei modelli di AI da utilizzare per l'inferenza. I modelli sono registrati come artefatti binari.
* **Swift (Object Storage):** utilizzato per l'archiviazione persistente delle immagini analizzate. Ogni upload segue una naming convention specifica (`terminale_` o `UI_`) per distinguere la sorgente e un UUID univoco.


## Struttura del repository
* `app_ia.py`: interfaccia web interattiva realizzata con Streamlit.
* `cli_ia.py`: script per l'esecuzione da riga di comando.
* `.gitignore`: configurazione per escludere il modello pesant e file temporanei, mantenendo il repository leggero.

## Requisiti e installazione
1. Ubuntu 22.04 LTS (o superiore).
2. Una distribuzione DevStack attiva e funzionante.

Per integrare il progetto in OpenStack, aggiungere la seguente riga al file local.conf in DevStack:
enable_plugin progetto-ia https://github.com/tuo-utente/progetto-ia

Successivamente, esegui ./stack.sh. Il plugin installerà automaticamente le dipendenze Python necessarie

## Setup
Prima di avviare l'applicazione, è necessario configurare l'ambiente Cloud tramite CLI di OpenStack creando il porgetto e l'utente dedicato.

openstack project create --domain default IA_Lab
openstack user create --domain default --project IA_Lab --password tua_password utente_1
openstack role add --project IA_Lab --user utente_1 member

Successivamente, i modelli di AI devono essere caricati su Glance:

openstack image create "YOLO11-Model-v1" --file yolo11n.pt --disk-format raw --container-format bare --private
openstack image create "YOLO11-Small" --file yolo11s.pt --disk-format raw --container-format bare --private
openstack image create "YOLO11-Segmentation" --file yolo11n-seg.pt --disk-format raw --container-format bare --private
openstack image create "YOLO11-Classifier" --file yolo11n-cls.pt --disk-format raw --container-format bare --private

## Esecuzione
Crea un file user.env con i seguenti parametri (non è stato incluso  nel commit per sicurezza):

export OS_AUTH_URL=http://localhost/identity/v3
export OS_PROJECT_NAME=IA_Lab
export OS_USERNAME=utente_1
export OS_PASSWORD=tua_password
export OS_PROJECT_DOMAIN_NAME=Default
export OS_USER_DOMAIN_NAME=Default
export OS_IDENTITY_API_VERSION=3

Successivamente, avviare l'applicazione:

source user.env
streamlit run app_ia.py
