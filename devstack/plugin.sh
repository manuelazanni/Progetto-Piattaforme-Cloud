#!/bin/bash

# Funzione per l'installazione delle dipendenze
function install_ia_plugin {
    echo_summary "Installazione dipendenze Python per il Progetto IA"
    # Installiamo tutto il necessario via pip
    sudo pip3 install ultralytics streamlit python-dotenv stepic pillow
}

# Logica di esecuzione di DevStack
if is_service_enabled progetto-ia; then
    if [[ "$1" == "stack" && "$2" == "install" ]]; then
        install_ia_plugin
    fi
fi
