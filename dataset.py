import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets import Dataset as HFDataSet

from tokenizers import Tokenizer

class BilinguaDataset(Dataset):
    def __init__(self, ds: HFDataSet, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_tgt.token_to_id()])