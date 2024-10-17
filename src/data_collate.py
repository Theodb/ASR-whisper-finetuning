from dataclasses import dataclass
from typing import Any

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        # Separate input features and labels
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels and replace padding tokens with -100
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # Remove initial BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch