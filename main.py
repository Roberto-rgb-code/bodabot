import os
import io
import re
import json
import wave
import base64
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd
from fuzzywuzzy import fuzz
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# Cargar .env
# =========================
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

# =========================
# Config desde .env
# =========================
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT / "data.json")))
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
LANG_CODE = os.getenv("LANG_CODE", "es-MX")
TTS_VOICE = os.getenv("TTS_VOICE", "es-MX-Neural2-D")
TTS_VOICE_FALLBACK = os.getenv("TTS_VOICE_FALLBACK", "es-MX-Standard-C")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# =========================
# Google SDKs (STT/TTS)
# =========================
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech

# =========================
# FastAPI
# =========================
app = FastAPI(title="BodaBot API (JSON + Google NLP + Voz)", version="3.4-wedding-enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Memoria en runtime
# =========================
TABLES: Dict[str, List[Dict[str, Any]]] = {}
ALIASES: Dict[str, str] = {}
INVERTED: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
SCHEMA: Dict[str, List[Dict[str, str]]] = {}
SESSIONS: Dict[str, Dict[str, Any]] = defaultdict(dict)
DEFAULT_SESSION = "default"

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")
DATE_RE = re.compile(r"(\d{1,2})\s*(?:de)?\s*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s*(?:de)?\s*(\d{4})?", flags=re.IGNORECASE)

# =========================
# Schema for validation (from dbbodas.xlsx)
# =========================
SCHEMA_DEF = {
    "tblCategoriasProveedores": [
        {"column_name": "idCategoriaProveedor", "data_type": "int"},
        {"column_name": "nombreCategoria", "data_type": "varchar"},
        {"column_name": "descripcion", "data_type": "text"},
    ],
    "tblDieteticas": [
        {"column_name": "idDietetica", "data_type": "int"},
        {"column_name": "nomDietetica", "data_type": "varchar"},
    ],
    "tblEmpresasOrganizadores": [
        {"column_name": "idEmpresaOrganizador", "data_type": "int"},
        {"column_name": "nombreEmpresa", "data_type": "varchar"},
        {"column_name": "direccion", "data_type": "varchar"},
        {"column_name": "idEstado", "data_type": "int"},
        {"column_name": "idMunicipio", "data_type": "int"},
        {"column_name": "telefono", "data_type": "varchar"},
        {"column_name": "paginaWeb", "data_type": "varchar"},
    ],
    "tblEstados": [
        {"column_name": "idEstado", "data_type": "int"},
        {"column_name": "nombreEstado", "data_type": "varchar"},
    ],
    "tblItinerarios": [
        {"column_name": "idItinerario", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "horaInicio", "data_type": "date"},
        {"column_name": "horaFin", "data_type": "date"},
        {"column_name": "actividad", "data_type": "varchar"},
        {"column_name": "responsables", "data_type": "varchar"},
    ],
    "tblMensajesProveedores": [
        {"column_name": "idMensajeProveedor", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
        {"column_name": "tituloMensaje", "data_type": "varchar"},
        {"column_name": "Mensaje", "data_type": "varchar"},
    ],
    "tblMunicipios": [
        {"column_name": "idMunicipio", "data_type": "int"},
        {"column_name": "nombreMunicipio", "data_type": "varchar"},
        {"column_name": "idEstado", "data_type": "int"},
    ],
    "tblOfertasProveedores": [
        {"column_name": "idOfertaProveedor", "data_type": "int"},
        {"column_name": "oferta", "data_type": "decimal"},
        {"column_name": "estatus", "data_type": "bit"},
    ],
    "tblOrganizadoresBodas": [
        {"column_name": "idOrganizadorBoda", "data_type": "int"},
        {"column_name": "idEmpresaOrganizador", "data_type": "int"},
        {"column_name": "nombreOrganizador", "data_type": "varchar"},
        {"column_name": "email", "data_type": "varchar"},
        {"column_name": "tel", "data_type": "varchar"},
        {"column_name": "activo", "data_type": "bit"},
        {"column_name": "idUsuario", "data_type": "int"},
        {"column_name": "Password", "data_type": "varchar"},
        {"column_name": "usuarioID", "data_type": "int"},
    ],
    "tblPreguntasProveedores": [
        {"column_name": "idPreguntaProveedor", "data_type": "int"},
        {"column_name": "Pregunta", "data_type": "varchar"},
    ],
    "tblPresupuestos": [
        {"column_name": "idPresupuesto", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "idServicioCategoProv", "data_type": "int"},
        {"column_name": "costo", "data_type": "decimal"},
        {"column_name": "fechaPago", "data_type": "date"},
        {"column_name": "idStatusPago", "data_type": "int"},
        {"column_name": "descripcion", "data_type": "varchar"},
        {"column_name": "observaciones", "data_type": "varchar"},
        {"column_name": "idProductoProveedor", "data_type": "int"},
    ],
    "tblProductosProveedores": [
        {"column_name": "idProductoProveedor", "data_type": "int"},
        {"column_name": "idServicioProveedor", "data_type": "int"},
        {"column_name": "nombreProducto", "data_type": "varchar"},
        {"column_name": "precioProducto", "data_type": "decimal"},
        {"column_name": "descripcion", "data_type": "varchar"},
    ],
    "tblProveedores": [
        {"column_name": "idProveedor", "data_type": "int"},
        {"column_name": "nomNegocio", "data_type": "varchar"},
        {"column_name": "idCategoriaProveedor", "data_type": "int"},
        {"column_name": "sitioWeb", "data_type": "varchar"},
        {"column_name": "tel", "data_type": "varchar"},
        {"column_name": "email", "data_type": "varchar"},
        {"column_name": "codigoPostal", "data_type": "varchar"},
        {"column_name": "direccion", "data_type": "varchar"},
        {"column_name": "pais", "data_type": "varchar"},
        {"column_name": "idEstado", "data_type": "int"},
        {"column_name": "idMunicipio", "data_type": "int"},
        {"column_name": "precioInicial", "data_type": "decimal"},
        {"column_name": "usuarioID", "data_type": "int"},
        {"column_name": "password", "data_type": "varchar"},
    ],
    "tblProveedoresDieteticas": [
        {"column_name": "idProveedorDietetica", "data_type": "int"},
        {"column_name": "idDietetica", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
    ],
    "tblProveedoresMunicipios": [
        {"column_name": "idProveedorMunicipio", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
        {"column_name": "idMunicipio", "data_type": "int"},
    ],
    "tblRedesProveedores": [
        {"column_name": "idRedProveedor", "data_type": "int"},
        {"column_name": "idRedSocial", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
        {"column_name": "link", "data_type": "varchar"},
    ],
    "tblRedesSociales": [
        {"column_name": "idRedSocial", "data_type": "int"},
        {"column_name": "redSocial", "data_type": "varchar"},
        {"column_name": "estatus", "data_type": "bit"},
    ],
    "tblRespuestasProveedores": [
        {"column_name": "idRespuestaProveedor", "data_type": "int"},
        {"column_name": "idPreguntaProveedor", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
        {"column_name": "Respuesta", "data_type": "varchar"},
    ],
    "tblSeleccionarOfertas": [
        {"column_name": "idSeleccionOferta", "data_type": "int"},
        {"column_name": "idOfertaProveedor", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
    ],
    "tblSeleccionarOrganizadoresBodas": [
        {"column_name": "IdSeleccionarOrganizadorBoda", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "idOrganizadorBoda", "data_type": "int"},
    ],
    "tblServiciosCategoriasProvedores": [
        {"column_name": "idServicioCategoProv", "data_type": "int"},
        {"column_name": "idCategoriaProveedor", "data_type": "int"},
        {"column_name": "serviciosProveedores", "data_type": "varchar"},
    ],
    "tblServiciosProveedores": [
        {"column_name": "idServicioProveedor", "data_type": "int"},
        {"column_name": "idProveedor", "data_type": "int"},
        {"column_name": "idServiciosCategoProv", "data_type": "int"},
    ],
    "tblStatusPagos": [
        {"column_name": "idStatusPago", "data_type": "int"},
        {"column_name": "nomStatus", "data_type": "varchar"},
    ],
    "tblTipoUsuarios": [
        {"column_name": "tipoUsuarioID", "data_type": "int"},
        {"column_name": "nombreTipoUsuario", "data_type": "nvarchar"},
    ],
    "tblUsuarios": [
        {"column_name": "usuarioID", "data_type": "int"},
        {"column_name": "nombre", "data_type": "nvarchar"},
        {"column_name": "email", "data_type": "nvarchar"},
        {"column_name": "tipoUsuarioID", "data_type": "int"},
        {"column_name": "password", "data_type": "varchar"},
    ],
    "tblAcompanantes": [
        {"column_name": "idAcompanate", "data_type": "int"},
        {"column_name": "idInvitado", "data_type": "int"},
        {"column_name": "acompanante", "data_type": "varchar"},
        {"column_name": "confirmado", "data_type": "bit"},
        {"column_name": "mesa", "data_type": "int"},
    ],
    "tblBebidas": [
        {"column_name": "idBebida", "data_type": "int"},
        {"column_name": "bebida", "data_type": "varchar"},
    ],
    "tblBebidasInvitado": [
        {"column_name": "idBebidaInvitado", "data_type": "int"},
        {"column_name": "idBebidaNovios", "data_type": "int"},
        {"column_name": "idInvitado", "data_type": "int"},
    ],
    "tblBebidasNovios": [
        {"column_name": "idBebidaNovios", "data_type": "int"},
        {"column_name": "idBebida", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
    ],
    "tblBioNovios": [
        {"column_name": "idBioNovio", "data_type": "int"},
        {"column_name": "idGenero", "data_type": "int"},
        {"column_name": "biografia", "data_type": "text"},
        {"column_name": "idNovios", "data_type": "int"},
    ],
    "tblConfiguracion": [
        {"column_name": "idConfiguracion", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "requiereAutenticacion", "data_type": "bit"},
        {"column_name": "dominioExterno", "data_type": "bit"},
    ],
    "tblCuentasCorreo": [
        {"column_name": "idCuentaCorreo", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "smtpServidor", "data_type": "varchar"},
        {"column_name": "usuario", "data_type": "varchar"},
        {"column_name": "password", "data_type": "varchar"},
        {"column_name": "port", "data_type": "int"},
    ],
    "tblEnviosDiarios": [
        {"column_name": "idEnvio", "data_type": "int"},
        {"column_name": "idCuenta", "data_type": "int"},
        {"column_name": "cantidad", "data_type": "int"},
        {"column_name": "fecha", "data_type": "datetime"},
    ],
    "tblHistoriaNovios": [
        {"column_name": "idNoviosHistoria", "data_type": "int"},
        {"column_name": "fechaConocieron", "data_type": "datetime"},
        {"column_name": "conocieron", "data_type": "text"},
        {"column_name": "imgConocieron", "data_type": "varchar"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "fechaPropuesta", "data_type": "datetime"},
        {"column_name": "propuesta", "data_type": "text"},
        {"column_name": "imgPropuesta", "data_type": "varchar"},
        {"column_name": "fechaPrimeraCita", "data_type": "datetime"},
        {"column_name": "primeraCita", "data_type": "text"},
        {"column_name": "imgPrimeraCita", "data_type": "varchar"},
        {"column_name": "fechaPedida", "data_type": "datetime"},
        {"column_name": "pedida", "data_type": "text"},
        {"column_name": "imgPedida", "data_type": "varchar"},
        {"column_name": "fechaNovios", "data_type": "datetime"},
        {"column_name": "novios", "data_type": "text"},
        {"column_name": "imgNovios", "data_type": "varchar"},
    ],
    "tblIdioma": [
        {"column_name": "idIdioma", "data_type": "int"},
        {"column_name": "idioma", "data_type": "varchar"},
    ],
    "tblIntentosIngreso": [
        {"column_name": "idIntentoIngreso", "data_type": "int"},
        {"column_name": "usuario", "data_type": "varchar"},
        {"column_name": "pass", "data_type": "varchar"},
        {"column_name": "fechaHora", "data_type": "datetime"},
        {"column_name": "ingreso", "data_type": "bit"},
    ],
    "tblInvitados": [
        {"column_name": "idInvitado", "data_type": "int"},
        {"column_name": "apodo", "data_type": "varchar"},
        {"column_name": "nombre", "data_type": "varchar"},
        {"column_name": "aPaterno", "data_type": "varchar"},
        {"column_name": "aMaterno", "data_type": "varchar"},
        {"column_name": "boletos", "data_type": "int"},
        {"column_name": "idGenero", "data_type": "int"},
        {"column_name": "boletosConfirmados", "data_type": "int"},
        {"column_name": "correo", "data_type": "varchar"},
        {"column_name": "idTipoInvitado", "data_type": "int"},
        {"column_name": "vienePor", "data_type": "int"},
        {"column_name": "telefono", "data_type": "varchar"},
        {"column_name": "idTipoMenu", "data_type": "int"},
        {"column_name": "accedio", "data_type": "datetime"},
        {"column_name": "llegoWeb", "data_type": "datetime"},
        {"column_name": "contrasena", "data_type": "varchar"},
        {"column_name": "invitacionEnviada", "data_type": "bit"},
        {"column_name": "soloMisa", "data_type": "bit"},
        {"column_name": "asistira", "data_type": "bit"},
        {"column_name": "qrEnviado", "data_type": "bit"},
        {"column_name": "qrConfirmado", "data_type": "bit"},
        {"column_name": "asistioBoda", "data_type": "bit"},
        {"column_name": "mesa", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "mensaje", "data_type": "text"},
        {"column_name": "regalo", "data_type": "bit"},
        {"column_name": "mensajeRegalo", "data_type": "text"},
        {"column_name": "idIdioma", "data_type": "int"},
    ],
    "tblMensajesWA": [
        {"column_name": "idMensajeWA", "data_type": "int"},
        {"column_name": "idTipoInvitado", "data_type": "int"},
        {"column_name": "idIdioma", "data_type": "int"},
        {"column_name": "mensajeInicial", "data_type": "varchar"},
        {"column_name": "mensajeMesaRegalos", "data_type": "varchar"},
        {"column_name": "mensajeDiaD", "data_type": "varchar"},
    ],
    "tblMesasNovios": [
        {"column_name": "idMesaNovio", "data_type": "int"},
        {"column_name": "idMesaRegalo", "data_type": "int"},
        {"column_name": "numero", "data_type": "varchar"},
        {"column_name": "idNovios", "data_type": "int"},
    ],
    "tblMesasRegalos": [
        {"column_name": "idMesaRegalo", "data_type": "int"},
        {"column_name": "nombre", "data_type": "varchar"},
        {"column_name": "url", "data_type": "varchar"},
        {"column_name": "telefono", "data_type": "varchar"},
    ],
    "tblNovios": [
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "nombreNovia", "data_type": "varchar"},
        {"column_name": "aPNovia", "data_type": "varchar"},
        {"column_name": "aMNovia", "data_type": "varchar"},
        {"column_name": "nombreNovio", "data_type": "varchar"},
        {"column_name": "aPNovio", "data_type": "varchar"},
        {"column_name": "aMNovio", "data_type": "varchar"},
        {"column_name": "fechaRegistro", "data_type": "datetime"},
        {"column_name": "dominio", "data_type": "varchar"},
        {"column_name": "padrinoNombre", "data_type": "varchar"},
        {"column_name": "padrinoApellido", "data_type": "varchar"},
        {"column_name": "madrinaNombre", "data_type": "varchar"},
        {"column_name": "madrinaApellido", "data_type": "varchar"},
        {"column_name": "misaLugar", "data_type": "varchar"},
        {"column_name": "misaHora", "data_type": "varchar"},
        {"column_name": "misaFecha", "data_type": "date"},
        {"column_name": "recepcionLugar", "data_type": "varchar"},
        {"column_name": "recepcionHora", "data_type": "varchar"},
        {"column_name": "recepcionFecha", "data_type": "date"},
        {"column_name": "fiestaLugar", "data_type": "varchar"},
        {"column_name": "fiestaHora", "data_type": "varchar"},
        {"column_name": "fiestaFecha", "data_type": "date"},
        {"column_name": "password", "data_type": "varchar"},
        {"column_name": "correoRegistro", "data_type": "varchar"},
        {"column_name": "telRegistro", "data_type": "varchar"},
        {"column_name": "idPlantilla", "data_type": "int"},
        {"column_name": "activo", "data_type": "bit"},
        {"column_name": "papaNovioNombre", "data_type": "varchar"},
        {"column_name": "mamaNovioNombre", "data_type": "varchar"},
        {"column_name": "papaNoviaNombre", "data_type": "varchar"},
        {"column_name": "mamaNoviaNombre", "data_type": "varchar"},
        {"column_name": "facebookNovio", "data_type": "varchar"},
        {"column_name": "facebookNovia", "data_type": "varchar"},
        {"column_name": "twitterNovio", "data_type": "varchar"},
        {"column_name": "twitterNovia", "data_type": "varchar"},
        {"column_name": "instagramNovio", "data_type": "varchar"},
        {"column_name": "instagramNovia", "data_type": "varchar"},
        {"column_name": "validacionCorreo", "data_type": "bit"},
        {"column_name": "misaLatitud", "data_type": "varchar"},
        {"column_name": "misaLongitud", "data_type": "varchar"},
        {"column_name": "recepcionLatitud", "data_type": "varchar"},
        {"column_name": "recepcionLongitud", "data_type": "varchar"},
        {"column_name": "fiestaLatitud", "data_type": "varchar"},
        {"column_name": "fiestaLongitud", "data_type": "varchar"},
        {"column_name": "dominioExterno", "data_type": "bit"},
        {"column_name": "usuarioID", "data_type": "int"},
    ],
    "tblPlantillaBoletos": [
        {"column_name": "idPM-boletos", "data_type": "int"},
        {"column_name": "plantillaBoleto", "data_type": "varchar"},
        {"column_name": "ancho", "data_type": "decimal"},
        {"column_name": "alto", "data_type": "decimal"},
    ],
    "tblPlantillas": [
        {"column_name": "idPlantilla", "data_type": "int"},
        {"column_name": "nombre", "data_type": "varchar"},
    ],
    "tblPlantillasNovios": [
        {"column_name": "idPlantillasNovios", "data_type": "int"},
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "idPM-invitacion", "data_type": "int"},
        {"column_name": "idPW-invitacion", "data_type": "int"},
        {"column_name": "idPWA-invitacion", "data_type": "int"},
        {"column_name": "idPM-boletos", "data_type": "int"},
        {"column_name": "idPWA-boletos", "data_type": "int"},
        {"column_name": "idPM-urgente", "data_type": "int"},
        {"column_name": "idPWA-urgente", "data_type": "int"},
        {"column_name": "idPM-diaD", "data_type": "int"},
        {"column_name": "idPWA-diaD", "data_type": "int"},
    ],
    "tblTipoInvitado": [
        {"column_name": "idTipoInvitado", "data_type": "int"},
        {"column_name": "tipoInvitado", "data_type": "varchar"},
        {"column_name": "tituloCorreo", "data_type": "varchar"},
        {"column_name": "descripcion", "data_type": "text"},
        {"column_name": "archivoBienvenida", "data_type": "varchar"},
        {"column_name": "idNoviosTI", "data_type": "int"},
    ],
    "vwMesasBoletos": [
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "mesa", "data_type": "int"},
        {"column_name": "boletos", "data_type": "int"},
        {"column_name": "boletosConfirmados", "data_type": "int"},
    ],
    "vwReporteGeneral": [
        {"column_name": "idNovios", "data_type": "int"},
        {"column_name": "invitaciones", "data_type": "int"},
        {"column_name": "boletos", "data_type": "int"},
        {"column_name": "boletosConfirmados", "data_type": "int"},
        {"column_name": "accedio", "data_type": "int"},
        {"column_name": "invitacionEnviada", "data_type": "int"},
        {"column_name": "SoloMisa", "data_type": "int"},
        {"column_name": "Asistira", "data_type": "int"},
        {"column_name": "QREnviado", "data_type": "int"},
        {"column_name": "QRConfirmado", "data_type": "int"},
        {"column_name": "asistioBoda", "data_type": "int"},
    ],
}

def normalize(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # Fix common wedding-related typos
    s = s.replace("fotografos", "fotógrafos").replace("fotograf", "fotógrafos")
    s = s.replace("salon", "salón").replace("invitacion", "invitación")
    return s

def tokenize(val: Union[str, int, float, None]) -> List[str]:
    if val is None:
        return []
    return [normalize(t) for t in WORD_RE.findall(str(val))]

def parse_date(text: str) -> Optional[datetime]:
    months = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    match = DATE_RE.search(text)
    if match:
        day = int(match.group(1))
        month = months[normalize(match.group(2))]
        year = int(match.group(3) or datetime.now().year)
        return datetime(year, month, day)
    return None

def validate_row(row: Dict[str, Any], table_name: str) -> bool:
    if not row or all(v is None for v in row.values()):
        return False
    schema = SCHEMA.get(table_name, [])
    for col in schema:
        col_name = col["column_name"]
        if col_name in row and row[col_name] is not None:
            if col["data_type"] == "int" and not isinstance(row[col_name], (int, float)):
                return False
            if col["data_type"] == "decimal" and not isinstance(row[col_name], (int, float)):
                return False
            if col["data_type"] in ("varchar", "text", "nvarchar") and not isinstance(row[col_name], str):
                return False
            if col["data_type"] == "bit" and not isinstance(row[col_name], (bool, int)):
                return False
            if col["data_type"] in ("datetime", "date") and not isinstance(row[col_name], str):
                return False
    return True

def _register_table(canon: str, rows: List[Any]) -> None:
    cleaned_rows = []
    for row in rows:
        if isinstance(row, dict) and validate_row(row, canon):
            cleaned_row = {k: v for k, v in row.items() if v is not None}
            cleaned_rows.append(cleaned_row)
    if cleaned_rows:
        TABLES[canon] = cleaned_rows
        ALIASES[normalize(canon)] = canon
        short = canon.split(".")[-1]
        ALIASES[normalize(short)] = canon
        no_tbl = re.sub(r"^tbl", "", short, flags=re.IGNORECASE)
        if no_tbl:
            ALIASES[normalize(no_tbl)] = canon
        singular = re.sub(r"(es|s)$", "", no_tbl, flags=re.IGNORECASE)
        if singular and normalize(singular) not in ALIASES:
            ALIASES[normalize(singular)] = canon

def _walk_and_collect(obj: Any, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            nxt = f"{prefix}.{k}" if prefix else k
            if isinstance(v, list):
                _register_table(nxt, v)
            else:
                _walk_and_collect(v, nxt)
    elif isinstance(obj, list):
        _register_table(prefix, obj)

def _build_inverted() -> None:
    INVERTED.clear()
    for tname, rows in TABLES.items():
        for i, row in enumerate(rows):
            tokens = set()
            for k, v in row.items():
                if k in ("nombreCategoria", "nomNegocio", "actividad", "descripcion", "tipoInvitado", "tituloCorreo", "nombreNovia", "nombreNovio", "misaLugar", "recepcionLugar"):
                    tokens.update(tokenize(v))
                else:
                    tokens.update(tokenize(v))
            for tok in tokens:
                INVERTED[tok].append((tname, i))

def load_data() -> None:
    if not DATA_PATH.exists():
        raise RuntimeError(f"No se encontró data.json en: {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error decodificando JSON: {e}")
    TABLES.clear()
    ALIASES.clear()
    SCHEMA.update(SCHEMA_DEF)
    _walk_and_collect(data)
    _build_inverted()

# Carga inicial
load_data()

def resolve_table(name: str) -> str:
    key = normalize(name)
    if key in ALIASES:
        return ALIASES[key]
    key2 = normalize(name.strip(' "\'')) if name else ""
    if key2 in ALIASES:
        return ALIASES[key2]
    for alias, canon in ALIASES.items():
        if fuzz.ratio(key, alias) > 80:
            return canon
    raise HTTPException(status_code=404, detail=f"Tabla '{name}' no encontrada. Revisa /tables")

# =========================
# RAG con ranking optimizado
# =========================
def rank_preferred_tables() -> List[str]:
    weights = {
        "proveedores": 9, "categoriasproveedores": 8, "itinerarios": 8,
        "presupuestos": 8, "productosproveedores": 7, "organizadoresbodas": 7,
        "empresasorganizadores": 6, "tipoInvitado": 8, "mesasregalos": 6,
        "estados": 5, "municipios": 5, "dieteticas": 6, "proveedoresdieteticas": 6,
        "novios": 9, "invitados": 8, "mesasnovios": 7, "mensajeswa": 6,
    }
    scored = []
    for canon in TABLES.keys():
        base = canon.split(".")[-1].lower()
        ntbl = re.sub(r"^tbl", "", base)
        score = sum(v for k, v in weights.items() if k in ntbl)
        scored.append((score, canon))
    scored.sort(reverse=True)
    return [c for _, c in scored]

PREFERRED = rank_preferred_tables()

def retrieve_context(question: str, max_items: int = 40) -> Dict[str, Any]:
    q_tokens = tokenize(question)
    if not q_tokens:
        ctx = {}
        for t in PREFERRED[:8]:
            rows = TABLES.get(t, [])
            if rows:
                ctx[t] = rows[:min(10, max_items)]
        return ctx

    hits: List[Tuple[str, int]] = []
    for t in q_tokens:
        hits.extend(INVERTED.get(t, []))
    counts = Counter(hits)

    by_table: Dict[str, List[int]] = defaultdict(list)
    for (tname, idx), _ in counts.most_common():
        by_table[tname].append(idx)

    ctx: Dict[str, Any] = {}
    max_per_table = max(3, max_items // 5)
    for t in PREFERRED + list(by_table.keys()):
        if t not in by_table or t in ctx:
            continue
        rows = TABLES.get(t, [])
        if not rows:
            continue
        idxs = by_table[t][:max_per_table]
        ctx[t] = [rows[i] for i in idxs if 0 <= i < len(rows)]

    if not ctx:
        for t in PREFERRED[:8]:
            rows = TABLES.get(t, [])
            if rows:
                ctx[t] = rows[:min(10, max_items)]
    return ctx

# =========================
# Google NLP (REST con API key)
# =========================
LANGUAGE_ENDPOINT = "https://language.googleapis.com/v2/documents:analyzeEntities"
CLASSIFY_ENDPOINT = "https://language.googleapis.com/v2/documents:classifyText"

def gnlp_analyze_entities(text: str, language: str = "es") -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        return {"entities": []}
    payload = {
        "document": {"type": "PLAIN_TEXT", "language": language, "content": text},
        "encodingType": "UTF8"
    }
    url = f"{LANGUAGE_ENDPOINT}?key={GOOGLE_API_KEY}"
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        return {"entities": []}
    return r.json()

def gnlp_classify_text(text: str, language: str = "es") -> Dict[str, Any]:
    if not GOOGLE_API_KEY or len(text) < 20:
        return {"categories": []}
    payload = {"document": {"type": "PLAIN_TEXT", "language": language, "content": text}}
    url = f"{CLASSIFY_ENDPOINT}?key={GOOGLE_API_KEY}"
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        return {"categories": []}
    return r.json()

# =========================
# NLU: intents + follow-ups
# =========================
GREETING_WORDS = {"hola", "buenos dias", "buenas tardes", "buenas noches", "hey", "qué tal", "que tal"}
WHO_ARE_YOU = {"quien eres", "quién eres", "como te llamas", "cómo te llamas"}
HELP_WORDS = {"ayuda", "que puedes hacer", "qué puedes hacer", "ayudame", "ayúdame", "como funcionas", "cómo funcionas"}
FOLLOW_MORE = {"mas", "más", "siguiente", "otra", "otro", "muestrame mas", "muéstrame más"}
FOLLOW_PICK_REGEX = re.compile(r"\b(el|la)\s+(\d+)\b")
FOLLOW_CONTACT = {"telefono", "teléfono", "correo", "email", "sitio", "web", "pagina", "página"}
FOLLOW_DETAILS = {"detalles", "detalle", "por tipo", "por mesa"}

CATEGORY_HINTS = {
    "flor": "floristerías", "flores": "floristerías",
    "foto": "fotógrafos", "fotógraf": "fotógrafos",
    "banquete": "banquetes", "salon": "salones", "salón": "salones",
    "pastel": "pastelerías", "pasteler": "pastelerías",
    "bebida": "bebidas", "bebidas": "bebidas",
    "vestido": "vestuario", "novia": "vestuario", "novio": "vestuario",
    "invitacion": "invitaciones", "invitación": "invitaciones",
    "mobiliario": "mobiliarios",
}

def detect_intent_and_slots(text: str) -> Dict[str, Any]:
    t = normalize(text)
    intents = []
    slots = {"date": None, "entidades": [], "categorias": []}

    # Extract date if present
    date = parse_date(t)
    if date:
        slots["date"] = date

    # Saludos / Identidad / Ayuda
    if any(w in t for w in GREETING_WORDS):
        return {"intent": "greeting", "slots": slots}
    if any(w in t for w in WHO_ARE_YOU):
        return {"intent": "who_are_you", "slots": slots}
    if any(w in t for w in HELP_WORDS):
        return {"intent": "help", "slots": slots}

    # Follow-ups
    if any(w in t for w in FOLLOW_MORE):
        return {"intent": "follow_more", "slots": slots}
    m = FOLLOW_PICK_REGEX.search(t)
    if m:
        try:
            idx = int(m.group(2))
            return {"intent": "follow_pick", "slots": {"index": idx, **slots}}
        except:
            pass
    if any(w in t for w in FOLLOW_CONTACT):
        return {"intent": "follow_contact", "slots": slots}
    if any(w in t for w in FOLLOW_DETAILS):
        return {"intent": "detalles_invitados", "slots": slots}

    # Wedding-specific intents
    if any(k in t for k in ["proveedor", "proveedores", "floristerías", "fotógrafos", "banquetes", "salones", "pastelerías", "bebidas", "vestuario", "invitaciones", "mobiliarios"]):
        intents.append("buscar_proveedores")
    if any(k in t for k in ["itinerario", "cronograma", "timeline", "hora", "actividad", "actividades", "ceremonia", "recepción", "fiesta"]):
        intents.append("itinerario")
    if any(k in t for k in ["presupuesto", "costo", "gastar", "pagar", "precio", "cuanto cuesta"]):
        intents.append("presupuesto")
    if any(k in t for k in ["invitado", "invitados", "lista de invitados", "tipo invitado"]):
        intents.append("invitados")
    if any(k in t for k in ["ubicacion", "ubicación", "donde", "dónde", "direccion", "dirección", "lugar", "venue", "misa", "recepcion", "fiesta"]):
        intents.append("ubicacion_evento")

    # Handle multi-intent or default to general
    intent = intents[0] if intents else "general"
    if len(intents) > 1:
        intent = "multi_intent"

    ents = gnlp_analyze_entities(text).get("entities", [])
    cats = gnlp_classify_text(text).get("categories", [])
    slots["entidades"] = ents
    slots["categorias"] = cats
    slots["intents"] = intents
    return {"intent": intent, "slots": slots}

# =========================
# Helpers de negocio
# =========================
def find_table(name_like: str) -> Optional[str]:
    name_like = normalize(name_like)
    for k in TABLES.keys():
        base = normalize(k.split(".")[-1])
        if name_like in base or fuzz.ratio(name_like, base) > 80:
            return k
    return None

def map_category_name_to_id(q: str) -> Optional[int]:
    tcat = find_table("categoriasproveedores")
    if not tcat:
        return None
    qn = normalize(q)
    for hint, stem in CATEGORY_HINTS.items():
        if hint in qn:
            for c in TABLES[tcat]:
                if stem in normalize(c.get("nombreCategoria")):
                    return c.get("idCategoriaProveedor")
    best = None
    best_score = 0
    for c in TABLES[tcat]:
        name = normalize(c.get("nombreCategoria"))
        if not name:
            continue
        score = max(fuzz.ratio(qn, name), fuzz.partial_ratio(qn, name))
        if score > best_score and score > 70:
            best = c.get("idCategoriaProveedor")
            best_score = score
    return best

def pick_state_municipio_from_text(q: str) -> Tuple[Optional[int], Optional[int]]:
    id_estado = id_muni = None
    t_est = find_table("estados")
    if t_est:
        for e in TABLES[t_est]:
            n = normalize(e.get("nombreEstado"))
            if n and (n in normalize(q) or fuzz.ratio(n, normalize(q)) > 80):
                id_estado = e.get("idEstado")
                break
    t_mun = find_table("municipios")
    if t_mun:
        for m in TABLES[t_mun]:
            n = normalize(m.get("nombreMunicipio"))
            if n and (n in normalize(q) or fuzz.ratio(n, normalize(q)) > 80):
                id_muni = m.get("idMunicipio")
                if not id_estado:
                    id_estado = m.get("idEstado")
                break
    return id_estado, id_muni

def filter_proveedores(q: str) -> List[Dict[str, Any]]:
    tprov = find_table("proveedores")
    if not tprov:
        return []
    toks = set(tokenize(q))
    cat_id = map_category_name_to_id(q)
    id_estado, id_muni = pick_state_municipio_from_text(q)

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for row in TABLES[tprov]:
        if cat_id and row.get("idCategoriaProveedor") != cat_id:
            continue
        if id_estado and row.get("idEstado") != id_estado:
            continue
        if id_muni and row.get("idMunicipio") != id_muni:
            continue

        score = 0
        for field in ("nomNegocio", "direccion", "pais", "sitioWeb", "email", "descripcion"):
            v = (row.get(field) or "")
            field_toks = set(tokenize(v))
            score += len(field_toks.intersection(toks))
            if field == "nomNegocio" and v:
                score += fuzz.partial_ratio(q, normalize(v)) // 20
        if any(k in normalize(q) for k in ["telefono", "teléfono"]):
            if row.get("tel"):
                score += 2
        if any(k in normalize(q) for k in ["precio", "costo"]):
            if row.get("precioInicial") is not None:
                score += 2
        scored.append((score, row))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [r for _, r in scored[:10] if _ > 0]

def filter_itinerario(date: Optional[datetime]) -> List[Dict[str, Any]]:
    tit = find_table("itinerarios")
    if not tit or not TABLES[tit]:
        return []
    items = TABLES[tit]
    if date:
        filtered = []
        for item in items:
            start_date = item.get("horaInicio")
            if start_date:
                try:
                    item_date = datetime.strptime(start_date.split(" ")[0], "%Y-%m-%d")
                    if item_date.date() == date.date():
                        filtered.append(item)
                except ValueError:
                    continue
        return sorted(filtered, key=lambda x: x.get("horaInicio") or "")[:6]
    return sorted(items, key=lambda x: x.get("horaInicio") or "")[:6]

# =========================
# Composición de respuesta
# =========================
def compose_answer(intent: str, text: str, ctx: Dict[str, Any], session: Dict[str, Any]) -> str:
    last = session.get("last", {})
    slots = session.get("last_slots", {})
    date = slots.get("date") or parse_date(text)

    if intent == "greeting":
        return "¡Hola! Soy BodaBot, tu asistente para planear tu boda perfecta. Puedo ayudarte con proveedores, itinerario, invitados, presupuestos o ubicaciones. ¿Qué necesitas hoy?"
    if intent == "who_are_you":
        return "Soy BodaBot, tu ayudante para bodas. Busco proveedores, organizo itinerarios, gestiono listas de invitados y resumo presupuestos usando tu data.json. ¡Dime cómo te ayudo!"
    if intent == "help":
        return "Puedo ayudarte a:\n1. Encontrar proveedores (ej. 'floristería en Guerrero')\n2. Crear itinerarios (ej. 'cronograma del 15 de noviembre')\n3. Gestionar invitados (ej. 'lista de invitados por tipo')\n4. Revisar presupuestos (ej. 'costo total de la boda')\n5. Buscar ubicaciones (ej. 'lugar de la recepción')\n¿Qué quieres explorar?"

    if intent == "follow_more":
        if last.get("type") == "proveedores" and last.get("candidates"):
            shown = last.get("shown", 0)
            step = 4
            nxt = last["candidates"][shown:shown+step]
            if not nxt:
                return "No hay más proveedores. ¿Quieres filtrar por otra categoría o ubicación?"
            session["last"]["shown"] = shown + len(nxt)
            session["last"]["display"] = nxt
            return render_proveedores(nxt, prefix="Más proveedores para tu boda:\n")
        if last.get("type") == "invitados":
            return "No hay más detalles de invitados. ¿Quieres ver por mesa o tipo específico?"
        return "¿De qué quieres ver más: proveedores, itinerario, invitados, presupuesto?"

    if intent == "follow_pick":
        if last.get("type") == "proveedores" and last.get("display"):
            m = re.search(r"\b(el|la)\s+(\d+)\b", normalize(text))
            if m:
                idx = int(m.group(2))
                disp = last["display"]
                if 1 <= idx <= len(disp):
                    item = disp[idx-1]
                    return render_proveedor_detalle(item)
            return "No reconocí el número. Usa por ejemplo: 'el 2' para detalles."
        return "¿De qué lista quieres elegir un número?"

    if intent == "follow_contact":
        if last.get("type") == "proveedores" and last.get("display"):
            lines = []
            for i, it in enumerate(last["display"], start=1):
                tel = it.get("tel") or "No disponible"
                mail = it.get("email") or "No disponible"
                web = it.get("sitioWeb") or "No disponible"
                lines.append(f"{i}. {it.get('nomNegocio') or '(sin nombre)'} · Tel: {tel} · Email: {mail} · Web: {web}")
            return "Contactos de los últimos proveedores:\n" + "\n".join(lines)
        return "Primero busca proveedores con algo como 'floristería en Chilapa' y luego te doy los contactos."

    if intent == "detalles_invitados":
        tinv = find_table("invitados")
        ttype = find_table("tipoInvitado")
        if tinv and ttype and TABLES[tinv] and TABLES[ttype]:
            type_map = {t.get("idTipoInvitado"): t.get("tipoInvitado") for t in TABLES[ttype] if t.get("tipoInvitado")}
            guests_by_type = defaultdict(list)
            for guest in TABLES[tinv]:
                type_id = guest.get("idTipoInvitado")
                if type_id in type_map:
                    name = f"{guest.get('nombre', '')} {guest.get('aPaterno', '')} {guest.get('aMaterno', '')}".strip()
                    guests_by_type[type_map[type_id]].append(name or "(sin nombre)")
            
            lines = []
            for type_name, guests in guests_by_type.items():
                lines.append(f"- {type_name}: {len(guests)} invitado(s)")
                for i, guest in enumerate(guests[:5], 1):  # Limit to 5 per type
                    lines.append(f"  {i}. {guest}")
                if len(guests) > 5:
                    lines.append(f"  ...y {len(guests) - 5} más")
            session["last"] = {"type": "invitados", "total": len(TABLES[tinv])}
            return f"Detalles de invitados por tipo:\n" + "\n".join(lines) + "\n\n¿Quieres ver por mesa o más detalles?"
        return "No hay datos de invitados. ¿Subimos una lista?"

    if intent == "buscar_proveedores":
        provs = filter_proveedores(text)
        if not provs:
            return "No encontré proveedores con ese criterio. Prueba con 'fotógrafos en Chilapa' o 'bebidas para boda'."
        session["last"] = {
            "type": "proveedores",
            "candidates": provs,
            "shown": len(provs[:4]),
            "display": provs[:4]
        }
        return render_proveedores(provs[:4])

    if intent == "itinerario":
        items = filter_itinerario(date)
        if items:
            session["last"] = {"type": "itinerario", "items": items}
            lines = []
            for i, it in enumerate(items, 1):
                time = it.get("horaInicio") or "N/D"
                if isinstance(time, str) and time.startswith("202"):
                    time = time.split(" ")[0]
                lines.append(f"{i}. {it.get('actividad')} ({time}, Resp: {it.get('responsables') or 'N/D'})")
            return f"Itinerario de la boda ({date.strftime('%d/%m/%Y') if date else 'muestra'}):\n" + "\n".join(lines) + "\n\n¿Ajustamos horarios o añadimos actividades?"
        return f"No hay itinerario para {date.strftime('%d/%m/%Y') if date else 'esa fecha'}. ¿Creamos uno con la ceremonia y recepción?"

    if intent == "presupuesto":
        tpre = find_table("presupuestos")
        if tpre and TABLES[tpre]:
            tot = 0.0
            by_cat = defaultdict(float)
            tcat = find_table("categoriasproveedores")
            cat_map = {c["idCategoriaProveedor"]: c["nombreCategoria"] for c in TABLES.get(tcat, [])}
            for p in TABLES[tpre]:
                costo = float(p.get("costo") or 0.0)
                if date:
                    try:
                        pago_date = datetime.strptime(p.get("fechaPago", "").split(" ")[0], "%Y-%m-%d")
                        if pago_date.date() != date.date():
                            continue
                    except ValueError:
                        continue
                tot += costo
                cat_id = p.get("idServicioCategoProv")
                cat = cat_map.get(cat_id, "Otros")
                by_cat[cat] += costo
            session["last"] = {"type": "presupuesto", "items": TABLES[tpre][:8]}
            lines = [f"- {cat}: ${amt:,.2f} MXN" for cat, amt in by_cat.items()]
            return f"Presupuesto de la boda{f' ({date.strftime('%d/%m/%Y')})' if date else ''}:\n" + "\n".join(lines) + f"\n\nTotal: ${tot:,.2f} MXN\n¿Filtramos por categoría o fecha?"
        return "No hay datos de presupuesto. ¿Cuál es tu presupuesto objetivo?"

    if intent == "invitados":
        tinv = find_table("tipoInvitado")
        if tinv and TABLES[tinv]:
            total = len(TABLES[tinv])
            by_type = Counter(t.get("tipoInvitado") for t in TABLES[tinv] if t.get("tipoInvitado"))
            session["last"] = {"type": "invitados", "total": total}
            lines = [f"- {typ}: {cnt} invitado(s)" for typ, cnt in by_type.most_common(5)]
            return f"Total: {total} invitados\n" + "\n".join(lines) + "\n\n¿Quieres detalles por tipo o mesa?"
        return "No hay datos de invitados. ¿Subimos una lista?"

    if intent == "ubicacion_evento":
        tnov = find_table("novios")
        parts = []
        if tnov and TABLES[tnov]:
            n = TABLES[tnov][0]
            if n.get("misaLugar"):
                parts.append(f"Ceremonia: {n.get('misaLugar')} ({n.get('misaFecha')})")
            if n.get("recepcionLugar"):
                parts.append(f"Recepción: {n.get('recepcionLugar')} ({n.get('recepcionFecha')})")
            if n.get("fiestaLugar"):
                parts.append(f"Fiesta: {n.get('fiestaLugar')} ({n.get('fiestaFecha')})")
        if parts:
            session["last"] = {"type": "ubicacion", "parts": parts}
            return "Ubicación del evento:\n" + "\n".join(parts) + "\n¿Necesitas un mapa o direcciones?"
        return "No hay datos de ubicación. ¿Me das el lugar de la ceremonia, recepción o fiesta?"

    if intent == "multi_intent":
        responses = []
        for sub_intent in slots.get("intents", []):
            if sub_intent == "itinerario":
                items = filter_itinerario(date)
                if items:
                    lines = []
                    for i, it in enumerate(items, 1):
                        time = it.get("horaInicio") or "N/D"
                        if isinstance(time, str) and time.startswith("202"):
                            time = time.split(" ")[0]
                        lines.append(f"{i}. {it.get('actividad')} ({time}, Resp: {it.get('responsables') or 'N/D'})")
                    responses.append(f"Itinerario ({date.strftime('%d/%m/%Y') if date else 'muestra'}):\n" + "\n".join(lines))
                else:
                    responses.append(f"No hay itinerario para {date.strftime('%d/%m/%Y') if date else 'esa fecha'}.")
            elif sub_intent == "presupuesto":
                tpre = find_table("presupuestos")
                if tpre and TABLES[tpre]:
                    tot = 0.0
                    by_cat = defaultdict(float)
                    tcat = find_table("categoriasproveedores")
                    cat_map = {c["idCategoriaProveedor"]: c["nombreCategoria"] for c in TABLES.get(tcat, [])}
                    for p in TABLES[tpre]:
                        costo = float(p.get("costo") or 0.0)
                        if date:
                            try:
                                pago_date = datetime.strptime(p.get("fechaPago", "").split(" ")[0], "%Y-%m-%d")
                                if pago_date.date() != date.date():
                                    continue
                            except ValueError:
                                continue
                        tot += costo
                        cat_id = p.get("idServicioCategoProv")
                        cat = cat_map.get(cat_id, "Otros")
                        by_cat[cat] += costo
                    lines = [f"- {cat}: ${amt:,.2f} MXN" for cat, amt in by_cat.items()]
                    responses.append(f"Presupuesto{f' ({date.strftime('%d/%m/%Y')})' if date else ''}:\n" + "\n".join(lines) + f"\nTotal: ${tot:,.2f} MXN")
                else:
                    responses.append("No hay datos de presupuesto.")
        session["last"] = {"type": "multi_intent", "intents": slots.get("intents", [])}
        session["last_slots"] = slots
        return "\n\n".join(responses) + "\n\n¿Quieres más detalles sobre alguno?"

    session["last_slots"] = slots
    snippet = []
    for t, rows in list(ctx.items())[:3]:
        snippet.append(f"- {t}: {len(rows)} registros")
    s = "\n".join(snippet) if snippet else "(sin datos relevantes)"
    return f"¡Hola! BodaBot está listo para ayudarte con tu boda.\nPrueba con: 'floristerías en Guerrero', 'itinerario del 15 de noviembre', 'presupuesto de bebidas'.\n\nContexto disponible:\n{s}"

def render_proveedores(items: List[Dict[str, Any]], prefix: str = "Proveedores para tu boda:\n") -> str:
    tcat = find_table("categoriasproveedores")
    cat_by_id = {c.get("idCategoriaProveedor"): c.get("nombreCategoria") for c in TABLES.get(tcat, [])}
    lines = []
    for i, it in enumerate(items, start=1):
        cat = cat_by_id.get(it.get("idCategoriaProveedor"), "N/D")
        precio = f" · Precio inicial: ${it.get('precioInicial'):,.2f} MXN" if it.get("precioInicial") is not None else ""
        lines.append(
            f"{i}. {it.get('nomNegocio') or '(sin nombre)'}"
            f"{' · ' + cat if cat else ''}"
            f"{' · Tel: ' + it['tel'] if it.get('tel') else ''}"
            f"{' · Email: ' + it['email'] if it.get('email') else ''}"
            f"{precio}"
        )
    lines.append("\nDi 'el 2' para detalles, 'más' para más opciones, o 'teléfono' para contactos.")
    return prefix + "\n".join(lines)

def render_proveedor_detalle(it: Dict[str, Any]) -> str:
    tcat = find_table("categoriasproveedores")
    cat = next((c.get("nombreCategoria") for c in TABLES.get(tcat, []) if c.get("idCategoriaProveedor") == it.get("idCategoriaProveedor")), "N/D")
    detalle = [
        f"Nombre: {it.get('nomNegocio') or '(sin nombre)'}",
        f"Categoría: {cat}",
        f"Teléfono: {it.get('tel') or 'No disponible'}",
        f"Email: {it.get('email') or 'No disponible'}",
        f"Web: {it.get('sitioWeb') or 'No disponible'}",
        f"Dirección: {it.get('direccion') or 'No disponible'}",
        f"País: {it.get('pais') or 'No disponible'}",
        f"Precio inicial: ${it.get('precioInicial'):,.2f} MXN" if it.get("precioInicial") is not None else "Precio inicial: No disponible",
        "¿Quieres que agende un contacto o busque alternativas?"
    ]
    return "\n".join(detalle)

# =========================
# STT/TTS helpers
# =========================
def speech_client() -> speech.SpeechClient:
    return speech.SpeechClient()

def tts_client() -> texttospeech.TextToSpeechClient:
    return texttospeech.TextToSpeechClient()

def stt_webm_opus(content: bytes, language_code: str = LANG_CODE) -> str:
    cli = speech_client()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    resp = cli.recognize(config=config, audio=audio)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript

def stt_wav_linear16(content: bytes, language_code: str = LANG_CODE) -> str:
    cli = speech_client()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    resp = cli.recognize(config=config, audio=audio)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript

def pcm16le_to_wav_bytes(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buff = io.BytesIO()
    with wave.open(buff, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buff.getvalue()

def tts_mp3(text: str, language_code: str = LANG_CODE, voice_name: Optional[str] = None) -> bytes:
    cli = tts_client()
    voice_name = voice_name or TTS_VOICE_FALLBACK
    synthesis_in = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    out = cli.synthesize_speech(input=synthesis_in, voice=voice, audio_config=cfg)
    return out.audio_content

def tts_wav_linear16(text: str, language_code: str = LANG_CODE, voice_name: Optional[str] = None) -> bytes:
    cli = tts_client()
    voice_name = voice_name or TTS_VOICE
    synthesis_in = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000
    )
    out = cli.synthesize_speech(input=synthesis_in, voice=voice, audio_config=cfg)
    return pcm16le_to_wav_bytes(out.audio_content, sample_rate=16000, channels=1)

# =========================
# Schemas
# =========================
class AskRequest(BaseModel):
    question: str
    max_ctx_items: Optional[int] = 40
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    model: str
    answer: str
    question: str
    used_sections: List[str]
    intent: str
    session_id: str

# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return {
        "name": "BodaBot API",
        "version": "3.4-wedding-enhanced",
        "endpoints": [
            "/health", "/tables", "/table/{name}", "/search",
            "/ask", "/ask_audio", "/ask_audio_wav", "/refresh"
        ],
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "data_path": str(DATA_PATH),
        "has_data": DATA_PATH.exists(),
        "tables_count": len(TABLES),
        "sample_tables": list(TABLES.keys())[:10],
        "google_nlp_key_loaded": bool(GOOGLE_API_KEY),
        "gcp_key_in_use": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        "lang_code": LANG_CODE,
        "tts_voice": TTS_VOICE,
        "cors_origins": CORS_ORIGINS,
    }

@app.post("/refresh")
def refresh():
    load_data()
    global PREFERRED
    PREFERRED = rank_preferred_tables()
    SESSIONS.clear()
    return {"reloaded": True, "tables": list(TABLES.keys())}

@app.get("/tables")
def tables():
    return {"tables": list(TABLES.keys())}

@app.get("/table/{name}")
def get_table(
    name: str,
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    canon = resolve_table(name)
    rows = TABLES.get(canon, [])
    return {"table": canon, "total": len(rows), "items": rows[offset: offset + limit]}

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    table: Optional[str] = Query(None),
    max_hits: int = Query(50, ge=1, le=1000),
):
    toks = tokenize(q)
    seen = set()
    results = []
    allowed = resolve_table(table) if table else None

    for t in toks:
        for tname, idx in INVERTED.get(t, []):
            if allowed and tname != allowed:
                continue
            if (tname, idx) in seen:
                continue
            seen.add((tname, idx))
            rows = TABLES.get(tname, [])
            if 0 <= idx < len(rows):
                results.append({"table": tname, "index": idx, "item": rows[idx]})
            if len(results) >= max_hits:
                break
    return {"query": q, "count": len(results), "results": results}

def _get_session(session_id: Optional[str], session_header: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    sid = session_id or session_header or DEFAULT_SESSION
    return sid, SESSIONS[sid]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(req.session_id, x_session_id)
    nlu = detect_intent_and_slots(req.question)
    ctx = retrieve_context(req.question, req.max_ctx_items or 40)
    answer = compose_answer(nlu["intent"], req.question, ctx, session)
    session["last_intent"] = nlu["intent"]
    session["last_slots"] = nlu["slots"]
    return AskResponse(
        model="google-nlp",
        answer=answer,
        question=req.question,
        used_sections=list(ctx.keys()),
        intent=nlu["intent"],
        session_id=sid,
    )

@app.post("/ask_audio")
async def ask_audio(audio: UploadFile = File(...), language: str = LANG_CODE, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(None, x_session_id)
    raw = await audio.read()
    user_text = stt_webm_opus(raw, language_code=language)
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="No se pudo transcribir el audio.")
    nlu = detect_intent_and_slots(user_text)
    ctx = retrieve_context(user_text, 40)
    answer = compose_answer(nlu["intent"], user_text, ctx, session)
    session["last_intent"] = nlu["intent"]
    session["last_slots"] = nlu["slots"]
    mp3 = tts_mp3(answer, language_code=language)
    return {
        "texto_usuario": user_text,
        "respuesta_texto": answer,
        "audio_base64": base64.b64encode(mp3).decode("utf-8"),
        "mime": "audio/mpeg",
        "used_sections": list(ctx.keys()),
        "intent": nlu["intent"],
        "session_id": sid,
    }

@app.post("/ask_audio_wav")
async def ask_audio_wav(audio: UploadFile = File(...), language: str = LANG_CODE, x_session_id: Optional[str] = Header(None)):
    sid, session = _get_session(None, x_session_id)
    try:
        raw = await audio.read()
        # Validate WAV file
        try:
            with wave.open(io.BytesIO(raw), 'rb') as wf:
                nchannels, sampwidth, framerate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
                if nchannels != 1 or framerate != 16000 or sampwidth != 2:
                    raise HTTPException(status_code=400, detail="El audio WAV debe ser mono, 16 kHz, 16-bit.")
        except wave.Error as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar el archivo WAV: {str(e)}")

        # Transcribe audio to text
        user_text = stt_wav_linear16(raw, language_code=language)
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo transcribir el audio (WAV).")

        # Process intent and context
        nlu = detect_intent_and_slots(user_text)
        ctx = retrieve_context(user_text, max_items=40)
        answer = compose_answer(nlu["intent"], user_text, ctx, session)

        # Update session
        session["last_intent"] = nlu["intent"]
        session["last_slots"] = nlu.get("slots", {})

        # Synthesize response to WAV
        wav_bytes = tts_wav_linear16(answer, language_code=language)

        return {
            "texto_usuario": user_text,
            "respuesta_texto": answer,
            "audio_wav_base64": base64.b64encode(wav_bytes).decode("utf-8"),
            "mime": "audio/wav",
            "used_sections": list(ctx.keys()),
            "intent": nlu["intent"],
            "session_id": sid,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud de audio: {str(e)}")