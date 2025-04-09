import os
import json
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from scipy.io import wavfile
import whisper
from utils import load_video_feats, add_noise  # Assuming these are available in your utils module
import sys
import uvicorn
import gdown

# Force DBG = False by simulating an argument
sys.argv.append("--start")

# Ensure the AV-HuBERT directory is in sys.path
av_hubert_dir = os.path.abspath("av_hubert/avhubert")
sys.path.append(av_hubert_dir)

app = FastAPI()

# Global configuration
model_type = "small"
device = "cuda" if torch.cuda.is_available() else "cpu"
use_av_hubert_encoder = 1
av_fusion = "separate"
whisper_path = "models/"
av_hubert_path = "av_hubert/avhubert/"
av_hubert_ckpt = "models/large_noise_pt_noise_ft_433h_only_weights.pt"
SAMPLE_RATE = 16000

# Model download logic
WHISPER_MODEL_URL = "https://drive.google.com/uc?id=15HVr--vidDSE1AYs_VvlMSx4o77r6dvp"
AV_HUBERT_MODEL_URL = "https://drive.google.com/uc?id=15WlAs3HIg7Xp87RDcpSZ3Bzg_qiRLJHM"

def download_model_if_missing(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(url, path, quiet=False)

# Load the Whisper-Flamingo model (defined BEFORE it's called)
def load_model(language="en", modalities="avsr", checkpoint_path=None, fp16=0):
    print(f"Loading model with checkpoint: {checkpoint_path}")
    try:
        model = whisper.load_model(
            model_type,
            download_root=whisper_path,
            video=True if modalities in ["avsr", "vsr"] else False,
            video_model_path=av_hubert_ckpt,
            av_hubert_path=av_hubert_path,
            av_hubert_encoder=use_av_hubert_encoder,
            av_fusion=av_fusion,
            add_gated_x_attn=1 if av_fusion == "separate" else 0
        )
    except Exception as e:
        print(f"Error loading model at whisper.load_model: {e}")
        raise
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = state_dict.get('state_dict', state_dict)
        state_dict_updated = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')} or state_dict
        try:
            model.load_state_dict(state_dict_updated)
        except RuntimeError:
            model.load_state_dict(state_dict_updated, strict=False)
    if device == "cuda" and fp16:
        model = model.cuda().half()
    else:
        model = model.to(device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, task="transcribe")
    return model, tokenizer

# Download and load models globally (AFTER load_model is defined)
download_model_if_missing("models/whisper-flamingo_en-x_small.pt", WHISPER_MODEL_URL)
download_model_if_missing("models/large_noise_pt_noise_ft_433h_only_weights.pt", AV_HUBERT_MODEL_URL)
model, tokenizer = load_model()  # This should now work

# Model request parameters
class TranscriptionRequest(BaseModel):
    language: str = "en"
    noise_snr: int = 0
    task: str = "transcribe"
    modalities: str = "avsr"
    beam_size: int = 1
    fp16: int = 0
    checkpoint_path: Optional[str] = "models/whisper-flamingo_en-x_small.pt"
    noise_fn: Optional[str] = "noise/babble/muavic/test.tsv"

# Process media files
def process_media(audio_file_path, video_file_path, language, noise_snr, task, modalities, beam_size, fp16, checkpoint_path=None, noise_fn=None):
    global model, tokenizer  # Use the global instances
    task_type = 'translate' if task == 'X-En' else 'transcribe'
    options = whisper.DecodingOptions(
        task=task_type,
        language=language,
        fp16=(device == "cuda" and fp16),
        without_timestamps=True,
        beam_size=None if beam_size == 1 else beam_size,
    )

    # Load and preprocess audio
    if audio_file_path:
        sample_rate, wav_data = wavfile.read(audio_file_path)
        assert sample_rate == SAMPLE_RATE, f"Sample rate must be {SAMPLE_RATE} Hz"
        audio = wav_data.flatten().astype(np.float32) / 32768.0
        if noise_snr < 100 and noise_fn and os.path.exists(noise_fn):
            with open(noise_fn, 'r') as f:
                noise_files = [ln.strip() for ln in f.readlines()]
            audio = add_noise(wav_data, noise_files, noise_snr=noise_snr).flatten().astype(np.float32) / 32768.0
        audio = whisper.pad_or_trim(audio, length=SAMPLE_RATE * 30)
        n_mels = 80 if model_type != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        if device == "cuda" and fp16:
            mel = mel.half().cuda()
        elif device == "cuda":
            mel = mel.cuda()
    else:
        mel = None

    # Load and preprocess video
    if video_file_path and modalities in ["avsr", "vsr"]:
        video = load_video_feats(video_file_path, train=False)
        video = torch.tensor(video, dtype=torch.float32).unsqueeze(0)
        video = video.permute(0, 4, 1, 2, 3)
        if device == "cuda" and fp16:
            video = video.half().cuda()
        elif device == "cuda":
            video = video.cuda()
    else:
        video = None

    # Perform decoding
    with torch.no_grad():
        if modalities == "avsr":
            result = model.decode(mel, options, video)
        elif modalities == "asr":
            result = model.decode(mel, options, video, test_a=True)
        elif modalities == "vsr":
            result = model.decode(mel, options, video, test_v=True)
        else:
            raise ValueError(f"Unsupported modality: {modalities}")
    
    transcription = result[0].text
    return {"text": transcription}

# API endpoint
@app.post("/transcribe/")
async def transcribe_file(
    audio_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
    params: str = Form(...)
):
    params = TranscriptionRequest(**json.loads(params))
    
    audio_file_path = video_file_path = None
    if audio_file:
        audio_file_path = f"temp_audio_{audio_file.filename}"
        with open(audio_file_path, "wb") as f:
            f.write(await audio_file.read())
    if video_file:
        video_file_path = f"temp_video_{video_file.filename}"
        with open(video_file_path, "wb") as f:
            f.write(await video_file.read())
    
    try:
        result = process_media(
            audio_file_path,
            video_file_path,
            params.language,
            params.noise_snr,
            params.task,
            params.modalities,
            params.beam_size,
            params.fp16,
            params.checkpoint_path,
            params.noise_fn
        )
    finally:
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        if video_file_path and os.path.exists(video_file_path):
            os.remove(video_file_path)
    
    return result

# Update port handling for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)