# Krok 1: Základní image
# Použijeme oficiální, tenkou (slim) verzi Pythonu 3.10.
FROM python:3.10-slim

# Krok 2: Nastavení pracovního prostředí
# Vytvoříme složku /app uvnitř kontejneru a budeme v ní pracovat.
WORKDIR /app

# Krok 3: Instalace závislostí
# Nejdříve zkopírujeme jen soubor se závislostmi pro efektivní využití cache.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Krok 4: Zkopírování celé aplikace
# Zkopírujeme všechny vaše .py soubory, data, a další potřebné soubory.
COPY . .

# Krok 5: Finální příkaz pro spuštění Streamlit aplikace
# Tento příkaz je speciálně upravený pro prostředí jako Cloud Run.
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false
