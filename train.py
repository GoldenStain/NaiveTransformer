from typing import Generator, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import random_split

from datasets import load_dataset, Dataset as HFDataSet
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds: HFDataSet, lang: str) -> Generator[str, None, None]:
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config: Dict[str, Any], ds: HFDataSet, lang: str) -> Tokenizer:
    # .format: insert lang into the {} of the str
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK], [SOS], [EOS], [PAD]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# TODO: add type annotation when dataset.py is done.
def get_ds(config: Dict[str, Any]):
    ds_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split='train')
    
    # tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split train and validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])