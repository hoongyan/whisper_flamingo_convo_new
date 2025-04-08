FROM continuumio/miniconda3:4.10.3
WORKDIR /app

RUN apt-get update && apt-get install -y sox git wget build-essential coreutils && rm -rf /var/lib/apt/lists/*
RUN conda create -n whisper-flamingo python=3.8 -y && \
    conda install -n whisper-flamingo -c conda-forge ffmpeg==4.2.2 -y
SHELL ["conda", "run", "-n", "whisper-flamingo", "/bin/bash", "-c"]

COPY . .
RUN pip install pip==24.0 gdown uvicorn==0.29.0 && \
    cd muavic-setup && pip install -r requirements.txt && cd .. && \
    cd av_hubert && pip install -r updated_requirements.txt && cd fairseq && pip install --editable ./ && cd /app && \
    pip install -r requirements.txt
RUN mkdir -p models && \
    gdown --id 15WlAs3HIg7Xp87RDcpSZ3Bzg_qiRLJHM -O models/large_noise_pt_noise_ft_433h_only_weights.pt && \
    gdown --id 15HVr--vidDSE1AYs_VvlMSx4o77r6dvp -O models/whisper-flamingo_en-x_small.pt

EXPOSE 8000
CMD ["conda", "run", "-n", "whisper-flamingo", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]