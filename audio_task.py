import argparse
import subprocess
from pathlib import Path
from time import perf_counter

import imageio_ffmpeg
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import save_json

MODEL_NAME = "openai/whisper-tiny"


def read_audio(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
               "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"]
    process = subprocess.run(command, capture_output=True, check=True)
    audio = np.frombuffer(process.stdout, dtype="<f4").copy()
    if not len(audio):
        raise ValueError("Аудиофайл пуст")
    if len(audio) > 30 * 16000:
        raise ValueError("Для этого учебного примера нужна запись не длиннее 30 секунд")
    return audio


def transcribe(path, language="ru"):
    audio = read_audio(path)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    started = perf_counter()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        tokens = model.generate(**inputs, language=language, task="transcribe", max_new_tokens=256)
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    elapsed = perf_counter() - started
    duration = len(audio) / 16000
    return {"file": str(path), "language": language, "text": text,
            "audio_seconds": duration, "processing_seconds": elapsed,
            "rtf": elapsed / duration}


def main():
    parser = argparse.ArgumentParser(description="Распознавание речи до 30 секунд")
    parser.add_argument("file")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--output", default="results/audio.json")
    args = parser.parse_args()
    result = transcribe(args.file, args.language)
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
