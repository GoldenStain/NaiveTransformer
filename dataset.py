import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset

from tokenizers import Tokenizer

class BilinguaDataset(Dataset):
    def __init__(self, ds: HFDataset, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Because the special tokens are used to decode the target language, so here we use tokenizer_tgt instead of tokenizer_src
        self.sos_token = torch.Tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.ds)
    
    # BilinguaDataset[index] includes a pair of one src_lang sentence and one tgt_lang sentence
    def __getitem__(self, index) -> dict:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # get tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # add special tokens
        enc_pad_cnt = self.seq_len - len(enc_input_tokens) - 2 # SOS and EOS
        dec_pad_cnt = self.seq_len - len(dec_input_tokens) - 1 # only SOS

        # check
        if min(enc_pad_cnt, dec_pad_cnt) < 0:
            raise ValueError('Input sentence is too long')

        # process encoder_input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_pad_cnt, dtype=torch.int64)
            ],
            dim=0
        )

        # process decoder_input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_pad_cnt, dtype=torch.int64)
            ],
            dim=0
        )

        # process label (only add EOS token)
        label = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_pad_cnt, dtype=torch.int64)
            ],
            dim=0
        )

        # double check
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            # encoder_mask和decoder_mask这里unsqueeze出一个额外维度是为了在和实际输入进行运算的时候能直接广播
            # 我们的注意力输入：(B, num_heads, seq_len, head_dim)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len) -> broadcast
            "label": label, # label
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0