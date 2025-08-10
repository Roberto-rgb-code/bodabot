Crea un entorno:

python -m venv venv  
.\venv\Scripts\activate   

Instala las dependencias:
pip install --upgrade pip
pip install -r requirements.txt

pon el .env en el proyecto

ejecuta con:


uvicorn main:app --reload --port 8000