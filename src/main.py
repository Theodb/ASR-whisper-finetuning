# main.py
import argparse
import sys
from train import train
from eval import eval

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate Whisper-large-v3 model with LoRA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Sub-parser for training
    train_parser = subparsers.add_parser("train", help="Fine-tune Whisper model with LoRA and mixed_precision using Accelerate")
    train_parser.add_argument("--dataset_name", type=str, default='google/fleurs', help="Name of the Huggingface dataset (e.g., 'google/fleurs')")
    train_parser.add_argument("--language", type=str, default='fr_fr', help="Language selection, e.g., 'fr_fr' or 'fr'")
    train_parser.add_argument("--num_train_epochs", type=int, default=10)
    train_parser.add_argument("--train_batch_size", type=int, default=4)
    train_parser.add_argument("--learning_rate", type=float, default=5e-5)
    train_parser.add_argument("--output_dir", type=str, default="./whisper-fr-LoRA", help="Where to save the LoRA weights and processor")
    train_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    train_parser.add_argument("--max_input_length", type=float, default=8, help="Max length of input audio (in seconds) increase if you have enough memory")
    train_parser.add_argument("--early_stopping_patience", type=int, default=3, help="How many epochs to wait for improvement before stopping early")
    train_parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum improvement needed to reset early stopping patience")
    train_parser.add_argument("--debug", action='store_true', help="Run in debug mode (use a small subset of the data)")
    train_parser.add_argument("--debug_subset_size", type=int, default=24, help="Number of samples to use in debug mode")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--auth_token", type=str, default="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", help="Huggingface authentication token for specific datasets like CommonVoice")

    # Sub-parser for evaluation
    eval_parser = subparsers.add_parser("eval", help="Evaluate LoRA fine-tuned Whisper model")
    eval_parser.add_argument("--dataset_name", type=str, default='google/fleurs', help="Name of the Huggingface dataset")
    eval_parser.add_argument("--language", type=str, default='fr_fr', help="Language selection, e.g., 'fr_fr' or 'fr'")
    eval_parser.add_argument("--batch_size", type=int, default=4)
    eval_parser.add_argument("--model_dir", type=str, default="./whisper-fr-LoRA", help="Directory containing the saved model")
    eval_parser.add_argument("--max_input_length", type=float, default=8, help="Max length of input audio (in seconds)")
    eval_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    eval_parser.add_argument("--auth_token", type=str, default="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", help="Huggingface authentication token for specific datasets like CommonVoice")

    args = parser.parse_args() 

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        eval(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
