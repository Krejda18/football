# Krok 1: Vyberte si základní obraz
# Použijeme oficiální, tenkou verzi Pythonu. Zvolte verzi, kterou používáte pro vývoj.
FROM python:3.10-slim

# Krok 2: Nastavte pracovní adresář uvnitř kontejneru
WORKDIR /app

# Krok 3: Nainstalujte závislosti
# Kopírováním requirements.txt zvlášť využijete Docker cache pro zrychlení budoucích sestavení.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Krok 4: Zkopírujte zbytek aplikace
# Zkopíruje všechny vaše .py soubory, složky Data, AVG - hodnoty, logo atd.
COPY . .

# Krok 5: Spusťte aplikaci
# Tento příkaz použije produkční server Gunicorn pro spuštění vaší webové aplikace.
# Předpokládá, že v souboru 'app.py' máte proměnnou 'app' (např. app = Flask(__name__)).
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
