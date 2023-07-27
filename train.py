import os
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer

from lion_pytorch import Lion
from palm_rlhf_pytorch import PaLM
from accelerate import Accelerator

# Constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024
SAVE_EVERY = 5000  # Frequency to save model and optimizer checkpoints

# Helpers
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# Accelerator
accelerator = Accelerator()
device = accelerator.device

# Instantiate PaLM
model = PaLM(
    num_tokens=256,
    dim=512,
    depth=8,
    flash_attn=True
).to(device)

# Prepare WikiText-2 data using TorchText
tokenizer = get_tokenizer('basic_english')
train_dataset, val_dataset, test_dataset = WikiText2()
data_train = torch.tensor([tokenizer(item) for item in train_dataset], dtype=torch.long)
data_val = torch.tensor([tokenizer(item) for item in val_dataset], dtype=torch.long)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1]
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# Optimizer
optimizer = AdamW(model.palm_parameters(), lr=LEARNING_RATE)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# Training
scaler = GradScaler()

torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmark for faster training

# Autoload file paths
MODEL_CHECKPOINT_PATH = "model_checkpoint.pth"
OPTIMIZER_CHECKPOINT_PATH = "optimizer_checkpoint.pth"

# Check if there are saved checkpoints and load them if available
if os.path.exists(MODEL_CHECKPOINT_PATH) and os.path.exists(OPTIMIZER_CHECKPOINT_PATH):
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_PATH))

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        with autocast():
            loss = model(next(train_loader), return_loss=True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    scaler.scale(loss).backward()
    accelerator.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss = model(next(val_loader), return_loss=True)
            accelerator.print(f"validation loss: {val_loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        accelerator.print(f"%s \n\n %s", (prime, "*" * 100))

        with torch.no_grad():
            sample = model.generate(GENERATE_LENGTH, inp[None, ...])
        output_str = decode_tokens(sample[0])
        accelerator.print(output_str, "\n")

    # Save model and optimizer checkpoints
    if i % SAVE_EVERY == 0:
        torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
        torch.save(optimizer.state_dict(), OPTIMIZER_CHECKPOINT_PATH)
