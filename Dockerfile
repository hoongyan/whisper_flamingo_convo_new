FROM continuumio/miniconda3:4.10.3
WORKDIR /app
RUN apt-get update && apt-get install -y sox git wget && rm -rf /var/lib/apt/lists/*
RUN conda create -n whisper-flamingo python=3.8 -y && \
    conda install -n whisper-flamingo -c conda-forge ffmpeg==4.2.2 -y
SHELL ["conda", "run", "-n", "whisper-flamingo", "/bin/bash", "-c"]
COPY requirements.txt av_hubert/updated_requirements.txt .
COPY muavic/ muavic/
COPY muavic-setup/ muavic-setup/
COPY av_hubert/ av_hubert/
RUN mkdir -p models && \
    wget -O models/whisper-flamingo_en-x_small.pt "https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_small.pt" && \
    wget -O models/large_noise_pt_noise_ft_433h_only_weights.pt "https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt"
RUN cd muavic && pip install -r requirements.txt && cd ..
RUN cd muavic-setup && pip install -r requirements.txt && cd ..
RUN cd av_hubert && pip install -r /app/updated_requirements.txt && cd fairseq && pip install --editable ./ && cd ../..
RUN pip install -r requirements.txt
RUN pip install pip==24.0
COPY . .
EXPOSE 8000
CMD ["conda", "run", "-n", "whisper-flamingo", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]