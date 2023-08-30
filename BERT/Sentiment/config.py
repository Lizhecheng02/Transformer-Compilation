import transformers

DEVICE = 'cpu'
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 10
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = './Model/pytorch_model.bin'
TRAINING_FILE = './Data/IMDB Dataset.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
