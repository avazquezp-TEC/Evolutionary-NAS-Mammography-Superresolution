import gdown
import zipfile
import os

file_id = "1cJa_I6f3CyoyCOQMqXjbBzBkr35Uv-eG"

# Descargar en el directorio actual
gdown.download(
    id=file_id,
    output="dataset.zip",
    quiet=False
)

# Extraer en el directorio actual
with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall(".")
os.remove("dataset.zip")
