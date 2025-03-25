from transformers import DistilBertTokenizer, TFDistilBertModel, BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Constants and configuration
BERT_MODEL_TYPE = 'distilbert'  # Or 'bert-base-uncased', etc.
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
SAVED_MODELS_DIR = Path(f"../saved_models/{BERT_MODEL_TYPE}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAX_CHUNK_LENGTH = 512  # Consistent chunk length
CHUNK_OVERLAP_PERCENTAGE = 0.2


class Encoder:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertModel.from_pretrained(model_name)

    def encode(self, text, max_length=400):
        inputs = self.tokenizer(text, return_tensors="tf", max_length=max_length, truncation=True, add_special_tokens=True)
        outputs = self.model(inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


class BertLargeEncoder:
    def __init__(self, model_name='bert-large-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertModel.from_pretrained(model_name)

    def encode(self, text, max_length=400):
        inputs = self.tokenizer(text, return_tensors="tf", max_length=max_length, truncation=True, add_special_tokens=True,
                                padding="max_length")
        outputs = self.model(inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


class FileManager:
    @staticmethod
    def read_text_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' ')

    @staticmethod
    def save_embeddings(output_dir, tensor_embedding, malware_sample_names):
        np.save(os.path.join(output_dir, 'tensor_embedding.npy'), tensor_embedding.numpy())
        with open(os.path.join(output_dir, 'malware_sample_names.txt'), 'w') as file:
            for name in malware_sample_names:
                file.write("%s\n" % name)


class EncoderWithChunks:
    def __init__(self, model_name='distilbert-base-uncased', max_tokens=MAX_CHUNK_LENGTH):  # Use MAX_CHUNK_LENGTH
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertModel.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def _split_text_into_chunks(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_size = self.max_tokens - 2  # Reserve space for [CLS] and [SEP]
        stride = int(chunk_size * (1 - CHUNK_OVERLAP_PERCENTAGE))  # Use Overlap Percentage
        chunks = []
        num_tokens = len(tokens)
        start_positions = list(range(0, num_tokens, stride))
        for start in start_positions:
            end = min(start + chunk_size, num_tokens)  # prevent index error
            chunk_tokens = tokens[start:end]
            chunks.append(chunk_tokens)
        return chunks

    def encode(self, text):
        chunks = self._split_text_into_chunks(text)
        cls_embeddings = []
        for chunk_tokens in chunks:
            input_ids = [self.tokenizer.cls_token_id] + chunk_tokens + [self.tokenizer.sep_token_id]
            input_ids = input_ids[:self.max_tokens]
            attention_mask = [1] * len(input_ids)
            padding_length = self.max_tokens - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            input_ids = tf.constant([input_ids], dtype=tf.int32)
            attention_mask = tf.constant([attention_mask], dtype=tf.int32)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cls_embeddings.append(cls_embedding)

        final_embedding = tf.reduce_mean(tf.concat(cls_embeddings, axis=0), axis=0)
        return final_embedding


class EmbeddingScaler:
    @staticmethod
    def scale_embeddings(embeddings):
        scaled_embeddings = 2. * (embeddings - tf.reduce_min(embeddings)) / (
                tf.reduce_max(embeddings) - tf.reduce_min(embeddings)) - 1
        return scaled_embeddings


class Preprocessor:
    def __init__(self, encoder: Encoder, file_manager: FileManager, scaler: EmbeddingScaler):
        self.encoder = encoder
        self.file_manager = file_manager
        self.scaler = scaler

    def process_samples(self, input_dir, output_dir):
        embedding_list = []
        malware_sample_names = []

        for file in os.listdir(input_dir):
            sample_name = file.split(".")[0]
            malware_sample_names.append(sample_name)

            print(f"Processing {sample_name}...")

            file_path = os.path.join(input_dir, file)
            text = self.file_manager.read_text_from_file(file_path)
            cls_embedding = self.encoder.encode(text)
            scaled_embedding = self.scaler.scale_embeddings(cls_embedding)
            embedding_list.append(scaled_embedding)

        tensor_embedding = tf.concat(embedding_list, axis=0)
        self.file_manager.save_embeddings(output_dir, tensor_embedding, malware_sample_names)

        return tensor_embedding


if __name__ == '__main__':
    for malware_type in MALWARE_TYPES:  # Iterate over malware types
        input_dir = MALWARE_DIR / malware_type
        output_dir = SAVED_MODELS_DIR / malware_type  # Save to the correct directory
        output_dir.mkdir(parents=True, exist_ok=True)

        encoder = EncoderWithChunks()  # Use EncoderWithChunks
        file_manager = FileManager()
        scaler = EmbeddingScaler()
        preprocessor = Preprocessor(encoder, file_manager, scaler)

        embeddings = preprocessor.process_samples(str(input_dir), str(output_dir))  # Pass string paths
        print(f"Embeddings processed and saved for {malware_type}.")