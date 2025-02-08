from transformers import DistilBertTokenizer, TFDistilBertModel, BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import os

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
        inputs = self.tokenizer(text, return_tensors="tf", max_length=max_length, truncation=True, add_special_tokens=True, padding="max_length")
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
    def __init__(self, model_name='distilbert-base-uncased', max_tokens=512):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertModel.from_pretrained(model_name)
        self.max_tokens = max_tokens  # DistilBERT's maximum token limit

    def _split_text_into_chunks(self, text):
        # Tokenize without adding special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_size = self.max_tokens - 2  # Reserve space for [CLS] and [SEP]
        stride = chunk_size // 2  # 50% overlap
        chunks = []
        num_tokens = len(tokens)
        # Generate start positions for chunks
        start_positions = list(range(0, num_tokens, stride))
        for start in start_positions:
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(chunk_tokens)
            if end >= num_tokens:
                break
        return chunks

    def encode(self, text):
        # Split text into overlapping chunks
        chunks = self._split_text_into_chunks(text)

        cls_embeddings = []
        for chunk_tokens in chunks:
            # Add [CLS] and [SEP] tokens
            input_ids = [self.tokenizer.cls_token_id] + chunk_tokens + [self.tokenizer.sep_token_id]
            # Truncate to max_tokens if necessary
            input_ids = input_ids[:self.max_tokens]
            # Create attention mask
            attention_mask = [1] * len(input_ids)
            # Pad input_ids and attention_mask if necessary
            padding_length = self.max_tokens - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            # Convert to tensors
            input_ids = tf.constant([input_ids], dtype=tf.int32)
            attention_mask = tf.constant([attention_mask], dtype=tf.int32)
            # Get the model's outputs
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] embedding
            cls_embeddings.append(cls_embedding)

        # Compute the average of all [CLS] embeddings
        final_embedding = tf.reduce_mean(tf.concat(cls_embeddings, axis=0), axis=0)
        return final_embedding


class EmbeddingScaler:
    @staticmethod
    def scale_embeddings(embeddings):
        scaled_embeddings = 2. * (embeddings - tf.reduce_min(embeddings)) / (tf.reduce_max(embeddings) - tf.reduce_min(embeddings)) - 1
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


    # input_dir = os.path.join(os.getcwd(), 'data', 'Malicia (Big 3 - Opcodes)', 'zbot')
    # output_dir = os.path.join(os.getcwd(), 'distelBERT_embeddings', 'Malicia (Big 3 - Opcodes)', 'zbot')
    # os.makedirs(output_dir, exist_ok=True)

    # input_dir = os.path.join(os.getcwd(), 'data', 'Malicia (Big 3 - Opcodes)', 'zeroaccess')
    # output_dir = os.path.join(os.getcwd(), 'distelBERT_embeddings', 'Malicia (Big 3 - Opcodes)', 'zeroaccess')
    # os.makedirs(output_dir, exist_ok=True)

    # Generatng BERT Large embeddings and obtaining its cls token
    input_dir = os.path.join(os.getcwd(), 'data', 'Malicia (Big 3 - Opcodes)', 'winwebsec')
    output_dir = os.path.join(os.getcwd(), 'bert-large_embeddings', 'Malicia (Big 3 - Opcodes)', 'winwebsec')
    os.makedirs(output_dir, exist_ok=True)

    # encoder = Encoder()
    # encoder = BertLargeEncoder()
    encoder = EncoderWithChunks()
    file_manager = FileManager()
    scaler = EmbeddingScaler()
    preprocessor = Preprocessor(encoder, file_manager, scaler)

    embeddings = preprocessor.process_samples(input_dir, output_dir)
    print("Embeddings processed and saved.")