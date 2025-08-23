# Krok 1: Základní image
# Použijeme oficiální, tenkou (slim) verzi Pythonu 3.10. Můžete změnit podle potřeby.
FROM python:3.10-slim

# Krok 2: Nastavení pracovního prostředí
# Vytvoříme složku /app uvnitř kontejneru a budeme v ní pracovat.
WORKDIR /app

# Krok 3: Instalace závislostí
# Nejdříve zkopírujeme jen soubor se závislostmi. To zrychlí budoucí sestavení,
# protože tento krok se přeskočí, pokud se requirements.txt nezmění.
COPY requirements.txt requirements.txt

# Spustíme instalaci všech knihoven z requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Krok 4: Zkopírování celé aplikace
# Nyní zkopírujeme všechny ostatní soubory a složky do kontejneru.
# To zahrnuje app.py, core_logic.py, složky Data, AVG - hodnoty atd.
COPY . .

# Krok 5: Spuštění aplikace
# Tento příkaz spustí vaši aplikaci pomocí produkčního serveru Gunicorn.
# Předpokládá, že v souboru 'app.py' máte proměnnou 'app' (např. app = Flask(__name__)).
# Gunicorn se spustí na portu, který mu přidělí Cloud Run ($PORT).
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false

