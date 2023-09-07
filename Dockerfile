# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    curl \
    git \
    wget \
    bzip2 \
    libglib2.0-0 \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Download and install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh

# Add Conda to path
ENV PATH="/root/miniconda/bin:${PATH}"


# Initialize Conda (this is important, or else some Conda commands might not work)
RUN conda init bash

# Create a new Conda environment and install some packages
RUN conda create -n autoavsr python=3.8 && \
    echo "source activate autoavsr" > ~/.bashrc && \
    /bin/bash -c "source activate autoavsr"

RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN conda install -c conda-forge ffmpeg

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=test.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
