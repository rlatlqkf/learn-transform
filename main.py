import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re, os
from tqdm import tqdm

# Constants
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Dataset Class
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = re.sub(r"([?.!,])", r" ", turn["Q"])
        a = re.sub(r"([?.!,])", r" ", turn["A"])

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)

        q_len, a_len = len(q_toked), len(a_toked)
        if q_len + a_len > self.max_len:
            q_toked = q_toked[-(int(self.max_len / 2)):]
            a_toked = a_toked[:self.max_len - len(q_toked)]

        labels = [self.mask] * len(q_toked) + a_toked[1:]
        mask = [0] * len(q_toked) + [1] * len(a_toked) + [0] * (self.max_len - len(q_toked) - len(a_toked))
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels) + [self.tokenizer.pad_token_id] * (self.max_len - len(labels))
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked) + [self.tokenizer.pad_token_id] * (self.max_len - len(q_toked) - len(a_toked))
        return token_ids, np.array(mask), labels_ids

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

# Load Tokenizer & Model
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK
)
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
epochs = 10
learning_rate = 3e-5
batch_size = 32
Sneg = -1e18
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_and_save_model(csv_path):
    print(f"\nProcessing: {csv_path}")
    try:
        data = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding='cp949')
    
    data.dropna(subset=["A", "Q"], inplace=True)
    train_set = ChatbotDataset(data, max_len=40)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    
    for epoch in range(epochs):
        dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for token_ids, mask, label in dataloader:
            optimizer.zero_grad()
            token_ids, mask, label = token_ids.to(device), mask.to(device), label.to(device)
            out = model(token_ids).logits
            mask_3d = mask.unsqueeze(2).repeat_interleave(out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            avg_loss = loss.sum() / mask.sum()
            avg_loss.backward()
            optimizer.step()
    
    model_save_path = os.path.join(save_dir, f"{os.path.basename(csv_path).replace('.csv', '.pth')}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs
    }, model_save_path)
    print(f"Model saved at: {model_save_path}")

# Process all CSV files in the directory
data_dir = "./data"  # 폴더 경로를 지정하세요
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    train_and_save_model(csv_file)

print("All models have been trained and saved.")
