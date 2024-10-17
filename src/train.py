# Import necessary modules and libraries
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model
import evaluate
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils import count_parameters, compute_module_sizes
from data_collate import DataCollatorSpeechSeq2SeqWithPadding

from huggingface_hub import login

# Setting up logging to show information level messages
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Common transcript keys across datasets
TRANSCRIPT_KEYS = ["text", "sentence", "normalized_text", "transcript", "transcription"]

def initialize_accelerator():
    """
    Initialize the Accelerator with the desired settings.

    Returns:
        Accelerator: Configured Accelerator instance.
    """
    accelerator = Accelerator(
        mixed_precision='fp16',  # Enable mixed precision for faster training
        device_placement=True,
        log_with="all"  # Adjust based on your logging preference
    )
    return accelerator


def set_environment(accelerator, seed):
    """
    Set random seed and adjust backend settings for optimal performance.

    Args:
        accelerator (Accelerator): Accelerator instance.
        seed (int): Seed for reproducibility.
    """
    set_seed(seed)
    if accelerator.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logging.info(f"Training on: {accelerator.device}, using mixed precision: {accelerator.mixed_precision}")


def load_and_prepare_datasets(args, processor):
    """
    Load and preprocess training and validation datasets.

    Args:
        args (argparse.Namespace): Parsed arguments.
        processor (WhisperProcessor): Whisper processor for handling text normalization.

    Returns:
        dict: Preprocessed 'train' and 'validation' datasets.
    """
    logging.info("Loading dataset...")

    login(token=args.auth_token)

    datasets = {}
    datasets["train"] = load_dataset(
        args.dataset_name,
        args.language,
        split="train",
        trust_remote_code=True
    )
    datasets["validation"] = load_dataset(
        args.dataset_name,
        args.language,
        split="validation",
        trust_remote_code=True
    )

    # Keep only the audio and transcript columns
    columns_to_keep = ["audio"] + TRANSCRIPT_KEYS  
    remove_columns = [col for col in datasets["train"].features.keys() if col not in columns_to_keep]
    datasets["train"] = datasets["train"].remove_columns(remove_columns)
    datasets["validation"] = datasets["validation"].remove_columns(remove_columns)

    # Use a small subset if in debug mode
    if args.debug:
        logging.info(f"Debug mode: Using only {args.debug_subset_size} samples.")
        datasets["train"] = datasets["train"].select(range(min(args.debug_subset_size, len(datasets["train"]))))
        datasets["validation"] = datasets["validation"].select(range(min(args.debug_subset_size, len(datasets["validation"]))))
    
    # Adjust the audio to the required sampling rate
    sampling_rate = processor.feature_extractor.sampling_rate
    datasets["train"] = datasets["train"].cast_column("audio", Audio(sampling_rate=sampling_rate))
    datasets["validation"] = datasets["validation"].cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Preprocessing function for datasets
    def preprocess_function(batch):
        audio = batch["audio"]
        processed = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=batch["transcription"],
        )
        processed["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return processed

    logging.info("Preprocessing datasets...")
    datasets["train"] = datasets["train"].map(
        preprocess_function,
        num_proc=args.num_workers,
        remove_columns=["audio", "transcription"],
    )
    datasets["validation"] = datasets["validation"].map(
        preprocess_function,
        num_proc=args.num_workers,
        remove_columns=["audio", "transcription"],
    )

    # Filter out long audio samples
    def filter_length(batch):
        return [length < args.max_input_length for length in batch["input_length"]]
    
    logging.info("Filtering long audio samples...")
    datasets["train"] = datasets["train"].filter(
        filter_length,
        batched=True,
        batch_size=1000,
        num_proc=args.num_workers,
        desc="Filtering training set"
    )
    datasets["validation"] = datasets["validation"].filter(
        filter_length,
        batched=True,
        batch_size=1000,
        num_proc=args.num_workers,
        desc="Filtering validation set"
    )

    return datasets

def setup_model(processor):
    """
    Load the Whisper model and apply LoRA adapters to fine-tune it.

    Args:
        processor (WhisperProcessor): The processor used for handling the model inputs.

    Returns:
        PeftModel: The Whisper model with LoRA adapters applied.
    """
    logging.info("Loading Whisper-large-v3 model and applying LoRA...")

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "distil-whisper/distil-large-v3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Set up LoRA configuration
    lora_config = LoraConfig(
        inference_mode=False,
        r=16,  # Dimensionality of LoRA
        lora_alpha=32,  # Scaling factor for LoRA
        lora_dropout=0.05,  # Dropout to prevent overfitting
        target_modules=['q_proj', 'v_proj'],  # The parts of the model we’re modifying
        bias="none"  # No bias term for simplicity
    )

    # Add LoRA to the Whisper model
    model = get_peft_model(whisper_model, lora_config)
    
    # Check and log the model size and parameter counts
    count_parameters(model)
    module_sizes = compute_module_sizes(model)
    logging.info(f"\nModel size: {module_sizes[''] * 1e-9:.2f} GB\n")

    return model


def prepare_dataloaders(args, datasets, data_collator):
    """
    Create DataLoaders for the training and validation datasets.

    Args:
        args (argparse.Namespace): Parsed arguments containing data settings.
        datasets (dict): The preprocessed training and validation datasets.
        data_collator (DataCollatorSpeechSeq2SeqWithPadding): Handles padding for batches.

    Returns:
        tuple: (train_dataloader, validation_dataloader)
    """
    logging.info("Creating DataLoaders...")

    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True  # Randomize order of training samples
    )

    validation_dataloader = DataLoader(
        datasets["validation"],
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False  # No need to shuffle validation data
    )

    return train_dataloader, validation_dataloader


def setup_optimizer_scheduler(model, learning_rate):
    """
    Set up the optimizer and learning rate scheduler for training.

    Args:
        model (PeftModel): The model to be trained.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        tuple: (optimizer, scheduler)
    """
    logging.info("Setting up optimizer and learning rate scheduler...")

    # Train only the LoRA adapters, so we grab the parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # AdamW optimizer for weight decay and better generalization
    optimizer = torch.optim.AdamW(params=trainable_params, lr=learning_rate)

    # Set up a learning rate scheduler that reduces the LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize the validation loss
        factor=0.5,  # Halve the learning rate when triggered
        patience=2,  # How many epochs to wait before reducing
        verbose=True,  # Show updates
        min_lr=1e-6  # Set a floor for the learning rate
    )

    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, accelerator):
    """
    Run one epoch of training.

    Args:
        model (PeftModel): The model to train.
        dataloader (DataLoader): The training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        accelerator (Accelerator): Accelerator to handle distributed training.

    Returns:
        float: The average loss for this training epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # Progress bar to track training
    progress_bar = tqdm(dataloader, desc="Training", disable=not accelerator.is_local_main_process, leave=False)
    
    for batch in progress_bar:
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss = loss / accelerator.num_processes  # Account for distributed training

        # Backpropagation
        accelerator.backward(loss)

        # Clip gradients to avoid exploding gradients
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, processor, wer_metric, accelerator):
    """
    Validate the model on the validation set and calculate WER.

    Args:
        model (PeftModel): The trained model.
        dataloader (DataLoader): Validation DataLoader.
        processor (WhisperProcessor): Processor to decode predictions.
        wer_metric (evaluate.Metric): Metric to calculate Word Error Rate (WER).
        accelerator (Accelerator): Accelerator for distributed validation.

    Returns:
        tuple: (average validation loss, average WER)
    """
    model.eval()
    total_eval_loss = 0.0
    total_wer = 0.0

    progress_bar = tqdm(dataloader, desc="Validating", disable=not accelerator.is_local_main_process, leave=False)
    
    for batch in progress_bar:
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss = loss / accelerator.num_processes  # Normalize loss for distributed training
            total_eval_loss += loss.item()

            # Decode predictions and calculate WER
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

            wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
            total_wer += wer
    
    avg_eval_loss = total_eval_loss / len(dataloader)
    avg_wer = total_wer / len(dataloader)
    return avg_eval_loss, avg_wer


def save_model(model, processor, output_dir, accelerator):
    """
    Save the trained model and processor.

    Args:
        model (PeftModel): The fine-tuned model.
        processor (WhisperProcessor): The processor used with the model.
        output_dir (str): Directory where the model and processor should be saved.
        accelerator (Accelerator): Accelerator instance to ensure saving happens only once.
    """
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)  # Save only the LoRA adapters
        processor.save_pretrained(output_dir)
        logging.info(f"Model and processor saved to {output_dir}")

def train(args):
    """
    Main function to handle the entire training process, including model setup, training, validation, 
    and saving the model. 
    """

    # Initialize Accelerator for handling hardware optimizations
    accelerator = initialize_accelerator()

    # Set environment, including seed and backend settings
    set_environment(accelerator, args.seed)

    # Load the Whisper processor for handling data
    logging.info("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3", language="french", task="transcribe")

    # Load and preprocess datasets
    datasets = load_and_prepare_datasets(
        args,
        processor=processor,
    )

    # Set up the data collator for padding and batching
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load Whisper model with LoRA adapters applied
    model = setup_model(processor)

    # Prepare DataLoaders for training and validation
    train_dataloader, validation_dataloader = prepare_dataloaders(
        args,
        datasets=datasets,
        data_collator=data_collator
    )

    # Set up optimizer and learning rate scheduler
    optimizer, scheduler = setup_optimizer_scheduler(model, args.learning_rate)

    # Prepare all components for training with Accelerator (handles parallelization and optimization)
    model, optimizer, train_dataloader, validation_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, scheduler
    )

    # Make sure the output directory exists
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up the evaluation metric for Word Error Rate (WER)
    wer_metric = evaluate.load("wer")

    # Initialize variables for early stopping
    best_wer = float('inf')  # Set the best WER to a large value initially
    epochs_no_improve = 0  # Track how many epochs since the last improvement

    logging.info("Starting the training process...")

    # Loop through each epoch
    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs", disable=not accelerator.is_local_main_process, leave=False):
        logging.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

        # Train for one epoch
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, accelerator)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")

        # Validate the model to check performance
        avg_eval_loss, avg_wer = validate(model, validation_dataloader, processor, wer_metric, accelerator)
        logging.info(f"Validation loss: {avg_eval_loss:.4f}, WER: {avg_wer:.4f}")

        # Adjust the learning rate based on validation loss
        scheduler.step(avg_eval_loss)

        # Check if this is the best model so far
        if avg_wer < best_wer - args.early_stopping_min_delta:
            best_wer = avg_wer
            epochs_no_improve = 0
            # Save the model if it improved
            save_model(model, processor, args.output_dir, accelerator)
            logging.info(f"New best WER: {best_wer:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in WER for {epochs_no_improve} epoch(s).")
            # If no improvement after a few epochs, stop training early
            if epochs_no_improve >= args.early_stopping_patience:
                logging.info("Early stopping triggered. No significant improvement.")
                break

    logging.info("Training process completed.")
    if accelerator.is_main_process:
        logging.info(f"Model and LoRA adapters have been saved to {args.output_dir}. You can now evaluate the performance.")

