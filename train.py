from typing import Generator, Dict, Any

import subprocess
import torchmetrics

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter


from datasets import load_dataset, Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilinguaDataset, causal_mask
from models import build_transformer
from config import get_weights_file_path,  latest_weights_file_path, get_config

from tqdm import tqdm

from models import Transformer


def get_all_sentences(ds: HFDataset, lang: str) -> Generator[str, None, None]:
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config: Dict[str, Any], ds: HFDataset, lang: str) -> Tokenizer:
    # .format: insert lang into the {} of the str
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=['[UNK], [SOS], [EOS], [PAD]'], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config: Dict[str, Any]) -> tuple:
    # prepare the cache dir
    cache_dir = Path("~/datasets").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.get('random_seed', 42))
    ds_raw = load_dataset(
        'opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split='train', cache_dir=str(cache_dir))

    # tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split train and validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilinguaDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                               config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilinguaDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                             config['lang_src'], config['lang_tgt'], config['seq_len'])

    # get longest sentence
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source language: {max_len_src}')
    print(f'Max length of target language: {max_len_tgt}')

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    # We want to check the sentences one by one, so the batch_size here is 1
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: Dict[str, Any], src_vocab_len: int, tgt_vocab_len: int):
    # We will leave other parameters to default
    model = build_transformer(
        src_vocab_len, tgt_vocab_len, config['seq_len'], config['d_model'])
    return model


def greedy_decode(model: Transformer, encoder_input: torch.Tensor, encoder_mask: torch.Tensor, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len: int, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(encoder_input, encoder_mask)
    # Initialize the decoder input with the sos token
    # 这里填充索引而不是填充向量，因为embedding是模型工作的一部分，是模型的第一层
    decoder_input = torch.empty(1, 1).fill_(
        sos_idx).type_as(encoder_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(
            1)).type_as(encoder_input).to(device)

        # calculate the output
        output = model.decode(decoder_input, encoder_output,
                              encoder_mask, decoder_mask)

        # gen next token
        prob = model.project(output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(
                encoder_input).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model: nn.Module, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len: int, device, print_msg: callable, global_step: int, writer: SummaryWriter, num_examples: int = 2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        result = subprocess.check_output(
            ['stty', 'size'], stderr=subprocess.DEVNULL)
        _, console_width = result.decode().split()
        console_width = int(console_width)
    except:
        # If we can't get the console width, then we use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(
                device)  # (B, 1, 1, seq_len)

            # make sure the batch_size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask,
                                         tokenizer_src, tokenizer_tgt, max_len, device)  # type: torch.Tensor

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            # The detach() function may be redundant, because we already enable the no_grad decorator
            model_output_text = tokenizer_tgt.decode(
                model_output.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)

            # Print the source, expected and predicted text
            print_msg('-'*console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_output_text}")

            if count == num_examples:
                print_msg("-"*console_width)
                break

            # 这里的metric因为没有使用torchmetrics的update方法，所以每次算出来的值都只基于当前的batch
            # 如果需要综合所有batch的信息，那么需要把metric在循环外定义，并在每次循环时使用update方法
            if writer:
                # Evaluate the character error rate
                # Compute the char error rate
                metric = torchmetrics.CharErrorRate()
                cer = metric(predicted, expected)
                writer.add_scalar('char error rate', cer, global_step)

                # Compute the word error rate
                metric = torchmetrics.WordErrorRate()
                wer = metric(predicted, expected)
                writer.add_scalar('word error rate', wer, global_step)

                # Compute the BLEU score
                metric = torchmetrics.BLEUScore()
                bleu = metric(predicted, expected)
                writer.add_scalar('BLEU score', bleu, global_step)


def train_model(config: dict):
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider use it for training")
    device = torch.device(device)

    # The folder for weights
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)
    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, then load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']  # type: str
    model_filename = latest_weights_file_path(config) if preload == 'latest' else (
        get_weights_file_path(config, preload) if preload else None)
    if model_filename:
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id(
        '[PAD]'), label_smoothing=0.1).to(device)

    # start training
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # clear the grad
            optimizer.zero_grad(set_to_none=True)

            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(
                device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(
                device)  # (B, 1, seq_len, seq_len)

            # Run tensors through the transformer model
            encoder_output = model.encode(
                encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(
                # (B, seq_len, d_model)
                decoder_input, encoder_output, encoder_mask, decoder_mask)
            # (B, seq_len, vocab_size)
            proj_output = model.project(decoder_output)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # 梯度裁剪
            # Update the weights
            optimizer.step()

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                       config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    config = get_config()
    train_model(config)
