# Base con Miniconda
FROM continuumio/miniconda3

# ---------------------------
# Add GUI support for tkinter, matplotlib, etc.
RUN apt-get update && apt-get install -y \
    tk \
    python3-tk \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# ---------------------------

# Carpeta de trabajo
WORKDIR /app

# Copiamos primero el environment para aprovechar la caché
COPY environment.yml /tmp/environment.yml

# Crear el entorno conda
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Copiar el resto del proyecto
COPY . .

# Ejecutar SIEMPRE dentro del entorno cpu312
CMD ["conda", "run", "--no-capture-output", "-n", "cpu312", "python", "models/Deep_learning/MLP_classifier.py"]


# # Base image with Miniconda
# FROM continuumio/miniconda3

# # Set working directory inside container
# WORKDIR /app

# # Copy Conda environment file and all project files
# COPY environment.yml .
# COPY . .

# # Create Conda environment from the environment.yml file
# RUN conda env create -f environment.yml

# # Use the Conda environment by default ("base" or your environment name)
# SHELL ["conda", "run", "-n", "cpu312", "/bin/bash", "-c"]

# # Command to run your main script when the container starts
# CMD ["python", "models/Deep_learning/MLP_classifier.py"]




# ----------------------- DEFAULT INFO ---------------------------
# # For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3-slim

# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED=1

# # Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

# WORKDIR /app
# COPY . /app

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# # During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "models/Deep_learning/MLP_classifier.py"]
