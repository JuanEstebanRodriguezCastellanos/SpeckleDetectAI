import streamlit as st
from PIL import Image
from filtro import Filtro
from unetresnet import procesar
import cv2

# Configuración de la página
st.set_page_config(
    page_title="Suavizado y detección",
    page_icon="🤖",
    layout="centered",
)

# CSS personalizado
st.markdown("""
    <style>
        .stApp {
            background-color: #42305d;
            color: white;
        }
        .stMarkdown h1, .stMarkdown p {
            color: white;
        }
        .css-1d391kg {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Suavizado y detección 🤖</h1>", unsafe_allow_html=True)
st.markdown("---")

MODEL_PATH = "modelo50.pth" 

def descargar_modelo():
    url = "https://drive.google.com/uc?export=download&id=15rLHBPmfmN2XF8tSFjWs8M6gRV7cuSUl"
    if not os.path.exists(MODEL_PATH):
        os.makedirs("modelo", exist_ok=True)
        st.info("Descargando modelo, por favor espera...")
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Modelo descargado correctamente.")

descargar_modelo()

# Función para aplicar el filtro
def procesarImagen(image):
    filtro = Filtro()
    return procesar(filtro.aplicar(image))

# Subida y procesamiento de imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_copy = image.copy()

    with st.spinner("Procesando..."):
        resultado = procesarImagen(image_copy)

    # Mostrar imágenes
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen original", use_container_width=True)
    with col2:
        st.image(resultado, caption="Imagen procesada", use_container_width=True)
