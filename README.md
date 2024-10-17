# Fine-Tuning Whisper Model with LoRA

This repository contains scripts for fine-tuning the distilled [Whisper-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) model on a specific language using Low-Rank Adaptation (LoRA) and evaluating the fine-tuned model. The scripts are optimized to run efficiently on GPUs and are designed to function without the need to produce a final trained model.

## Table of Contents

- [Objectives](#objectives)
- [Methodology](#methodology)
  - [Dataset Selection](#dataset-selection)
  - [Training Script](#training-script)
  - [Evaluation Script](#evaluation-script)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Methodology

### Dataset Selection

For testing and demonstration purposes, the **French Fleurs** dataset was used. This dataset consists of approximately 14 hours of audio recordings and their corresponding transcripts, making it suitable for quick testing and validation of the code.

The **French Fleurs** dataset includes:

- Reading-style audio recordings.
- Transcriptions in French.
- A manageable dataset size for testing (14 hours).

While the Fleurs dataset was useful for testing, it would be more suitable to run the model on larger and more domain-specific datasets. Potential datasets include:

- **Common Voice 17.0**: A large collection of voice data from volunteers worldwide.
- **VoxPopuli**: Multilingual speech corpus with extensive French recordings.
- **PORTMEDIA (French)**: Acted telephone dialogues suitable for call center scenarios.
- **TCOF (Adults)**: Spontaneous speech dataset from adult speakers.

The code is flexible and can be adapted to train on other Automatic Speech Recognition (ASR) datasets from Hugging Face. Ensure that the `--language` argument matches the dataset's language code. It would be interesting to run the script on **VoxPopuli** or **Common Voice** for more extensive training.

### Training Script

The training script leverages the Whisper-large-v3 model and fine-tunes it using the LoRA technique, which reduces the number of trainable parameters and speeds up training.

Key features:

- **Dataset Flexibility**: Can work with any Hugging Face ASR dataset. Adjust the `--dataset_name` and `--language` arguments accordingly.
- **Data Preprocessing**: Processes audio data and transcripts, filtering out long audio samples to optimize training time.
- **Model Adaptation**: Applies LoRA adapters to the Whisper model to enable efficient fine-tuning.
- **Optimization**: Employs mixed-precision training (`fp16`) and the `Accelerate` library for efficient GPU utilization.
- **Training Loop**: Incorporates early stopping based on validation performance to prevent overfitting.

### Evaluation Script

The evaluation script measures the performance of the fine-tuned model on a validation dataset.

Key features:

- **Data Loading**: Loads the evaluation dataset for the specified language.
- **Model Loading**: Loads the fine-tuned model along with the processor.
- **Metric Calculation**: Computes the Word Error Rate (WER) to evaluate transcription accuracy.
- **Batch Processing**: Processes data in batches for efficient GPU utilization.

## Requirements

- Python 3.7 or higher
- GPU with CUDA support
- Python packages listed in `requirements.txt`
- Hugging Face account and access token (for certain datasets and models)

## Usage

### Training

Run the training script to fine-tune the Whisper model:

```bash
python main.py train --dataset_name google/fleurs --language fr_fr --num_train_epochs 10 --train_batch_size 4 --learning_rate 5e-5 --output_dir ./whisper-fr-LoRA --auth_token YOUR_HF_TOKEN
```

**Arguments:**

- `--dataset_name`: Name of the Hugging Face dataset (default: `'google/fleurs'`).
- `--language`: Language code (e.g., `'fr_fr'` for French). Ensure this matches the dataset's language.
- `--num_train_epochs`: Number of training epochs (default: `10`).
- `--train_batch_size`: Batch size for training (default: `4`).
- `--learning_rate`: Learning rate for the optimizer (default: `5e-5`).
- `--output_dir`: Directory to save the fine-tuned model and processor (default: `'./whisper-fr-LoRA'`).
- `--auth_token`: Hugging Face access token.

**Optional Arguments:**

- `--num_workers`: Number of worker threads for data loading (default: `4`).
- `--max_input_length`: Maximum length of input audio in seconds (default: `8`).
- `--early_stopping_patience`: Number of epochs to wait for improvement before early stopping (default: `3`).
- `--early_stopping_min_delta`: Minimum improvement needed to reset early stopping patience (default: `0.0`).
- `--debug`: Run in debug mode with a small subset of data.
- `--debug_subset_size`: Number of samples to use in debug mode (default: `24`).
- `--seed`: Random seed for reproducibility (default: `42`).

**Example:**

To train on the **Fleurs** dataset in French:

```bash
python main.py train --dataset_name google/fleurs --language fr --num_train_epochs 5 --train_batch_size 8 --learning_rate 5e-5 --output_dir ./whisper-fr-LoRA --auth_token YOUR_HF_TOKEN
```

**Note:** Adjust the `--dataset_name` and `--language` parameters based on the dataset you choose.

### Evaluation

Run the evaluation script to assess the fine-tuned model:

```bash
python main.py eval --dataset_name google/fleurs --language fr_fr --batch_size 4 --model_dir ./whisper-fr-LoRA --auth_token YOUR_HF_TOKEN
```

**Arguments:**

- `--dataset_name`: Name of the Hugging Face dataset (default: `'google/fleurs'`).
- `--language`: Language code (e.g., `'fr_fr'` for French). Ensure this matches the dataset's language.
- `--batch_size`: Batch size for evaluation (default: `4`).
- `--model_dir`: Directory containing the saved model (default: `'./whisper-fr-LoRA'`).
- `--auth_token`: Hugging Face access token.

**Optional Arguments:**

- `--max_input_length`: Maximum length of input audio in seconds (default: `8`).
- `--num_workers`: Number of worker threads for data loading (default: `4`).

**Example:**

To evaluate the model on the **VoxPopuli** dataset in French:

```bash
python main.py eval --dataset_name mozilla-foundation/common_voice_17_0 --language fr --batch_size 4 --model_dir ./whisper-fr-LoRA --auth_token YOUR_HF_TOKEN
python main.py eval --dataset_name facebook/voxpopuli --language fr --batch_size 4 --model_dir ./whisper-fr-LoRA --auth_token YOUR_HF_TOKEN
```

## Project Structure

- `main.py`: Entry point script to run training and evaluation.
- `train.py`: Contains the training logic for fine-tuning the Whisper model.
- `eval.py`: Contains the evaluation logic to assess the fine-tuned model.
- `utils.py`: Utility functions for parameter counting and module size computation.
- `data_collate.py`: Custom data collator for speech sequence-to-sequence tasks with padding.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.
- `LICENSE`: License information.

## References

- [Whisper Model](https://github.com/openai/whisper)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Accelerate Library](https://huggingface.co/docs/accelerate/index)
- [Fleurs Dataset](https://huggingface.co/datasets/google/fleurs)
- [Common Voice Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- [VoxPopuli Dataset](https://huggingface.co/datasets/facebook/voxpopuli)
- **Other Potential Datasets**:
  - **PORTMEDIA (French)**: Acted telephone dialogues suitable for call center scenarios.
  - **TCOF (Adults)**: Spontaneous speech dataset from adult speakers.

