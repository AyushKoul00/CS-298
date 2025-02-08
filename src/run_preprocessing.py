from utils.Preprocessor import Encoder, BertLargeEncoder, EncoderWithChunks, FileManager, EmbeddingScaler, Preprocessor
import os

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

