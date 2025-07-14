import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import re

# ----------------------------
# Constants and Mappings
alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
int2dna.update({0: "<pad>"})
VALID_DNA_REGEX = re.compile(r'^[NACGT]+$', re.IGNORECASE)

# ----------------------------
# Model Definitions

class FixedLengthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 16, padding_idx=0)
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1]).squeeze(1)

class VariableLengthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 16, padding_idx=0)
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1]).squeeze(1)

# ----------------------------
# Utility Functions

def encode_sequence(seq, max_len=128):
    seq = seq.upper()
    int_seq = [dna2int.get(ch, 0) for ch in seq]
    # if len(int_seq) < max_len:
    #     int_seq += [0] * (max_len - len(int_seq))
    # else:
    #     int_seq = int_seq[:max_len]
    return torch.tensor([int_seq], dtype=torch.long)

def encode_variable_sequence(seq):
    seq = seq.upper()
    int_seq = [dna2int.get(ch, 0) for ch in seq]
    return torch.tensor(int_seq, dtype=torch.long).unsqueeze(0), torch.tensor([len(int_seq)])

# ----------------------------
# Load Models

@st.cache_resource
def load_models():
    fixed_model = FixedLengthModel()
    fixed_model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    fixed_model.eval()

    var_model = VariableLengthModel()
    var_model.load_state_dict(torch.load("model_unequal_length.pth", map_location=torch.device('cpu')))
    var_model.eval()

    return fixed_model, var_model

fixed_model, var_model = load_models()

# ----------------------------
# Streamlit UI

st.title("DNA CpG Counter - Model Inference")

user_input = st.text_input("Enter DNA Sequence (only characters A, C, G, T, N):")

if user_input:
    if not VALID_DNA_REGEX.match(user_input):
        st.error("Invalid DNA sequence. Please only use characters A, C, G, T, N.")
    else:
        with torch.no_grad():
            seq_len = len(user_input)
            if seq_len == 128:
                input_tensor = encode_sequence(user_input)
                prediction = fixed_model(input_tensor)
                st.success(f"Predicted CpG count (Fixed-length model): **{prediction.item():.2f}**")
            else:
                input_tensor, lengths = encode_variable_sequence(user_input)
                prediction = var_model(input_tensor, lengths)
                st.success(f"Predicted CpG count (Variable-length model): **{prediction.item():.2f}**")
