FROM continuumio/miniconda3:4.10.3
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y sox git wget build-essential && rm -rf /var/lib/apt/lists/*

# Set up Conda environment
RUN conda create -n whisper-flamingo python=3.8 -y && \
    conda install -n whisper-flamingo -c conda-forge ffmpeg==4.2.2 -y

# Use Conda shell
SHELL ["conda", "run", "-n", "whisper-flamingo", "/bin/bash", "-c"]

# Copy all files (includes muavic-setup/, av_hubert/, requirements.txt)
COPY . .

# Install muavic-setup dependencies
RUN cd muavic-setup && pip install -r requirements.txt && cd ..

# Downgrade pip
RUN pip install pip==24.0

# Install av_hubert and fairseq (no cloning, submodules assumed present)
RUN cd av_hubert && pip install -r updated_requirements.txt && cd fairseq && pip install --editable ./ && cd ../..

# Rest of your Dockerfile...
RUN mkdir -p models && \
    wget --progress=bar:force -O models/whisper-flamingo_en-x_small.pt "https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_small.pt" && \
    wget --progress=bar:force -O models/large_noise_pt_noise_ft_433h_only_weights.pt "https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt"
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["conda", "run", "-n", "whisper-flamingo", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]