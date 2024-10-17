#! python3.7

import argparse
import os
import numpy as np
import torch
import warnings

from datetime import datetime
from sys import platform
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
import evaluate
from tqdm.auto import tqdm

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper Model on Fleurs French Test Dataset")
    parser.add_argument("--model", default="large-v2", help="Model to use",
                        choices=[
                            "tiny",
                            "base",
                            "small",
                            "medium",
                            "large",
                            "large-v2",
                            "large-v2",
                            "large-v2"
                        ])
    parser.add_argument("--language", default="fr", help="Language code for transcription (e.g., 'fr' for French).")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for evaluation.")
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        fp16 = True
        print("Using NVIDIA GPU with CUDA.")
    elif torch.backends.mps.is_available():
        device = "mps"
        fp16 = True
        print("Using Apple Silicon MPS backend.")
    else:
        device = "cpu"
        fp16 = False
        print("Using CPU.")

    # Load the WER metric
    wer_metric = evaluate.load("wer")

    # Load the processor and model
    print(f"Loading Whisper model '{args.model}'...")
    try:
        model_path = f"openai/whisper-{args.model}"
        processor = WhisperProcessor.from_pretrained(model_path)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
        whisper_model.to(device)
        if fp16 and device != "cpu":
            whisper_model = whisper_model.half()
        language = args.language
        # Prepare forced_decoder_ids for French transcription
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    except Exception as e:
        print(f"Error loading Whisper model '{args.model}': {e}")
        return

    # Load the Fleurs French test dataset
    print("Loading Fleurs French test dataset...")
    fleurs = load_dataset("google/fleurs", "fr_fr", split="test")
    # Resample audio to 16000 Hz
    fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16000))

    # Prepare lists to store references and predictions
    references = []
    predictions = []

    # Process the dataset in batches
    batch_size = args.batch_size
    total_samples = len(fleurs)
    num_batches = (total_samples + batch_size - 1) // batch_size

    print("Starting evaluation...")
    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, total_samples)
        batch = fleurs.select(range(batch_start, batch_end))

        # Prepare input features
        audio_arrays = [sample["audio"]["array"] for sample in batch]
        inputs = processor(audio=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(device)
        if fp16 and device != "cpu":
            input_features = input_features.half()

        # Generate transcriptions
        with torch.no_grad():
            generated_ids = whisper_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=225,
                do_sample=False,
                num_beams=5,
            )
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Collect references and predictions
        references.extend([sample["transcription"] for sample in batch])
        predictions.extend(transcriptions)

    # Compute WER
    print("Computing WER...")
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"Average WER on the Fleurs French test set: {wer:.4f}")

if __name__ == "__main__":
    main()
