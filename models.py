import torch.nn as nn
import torch
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Why do we multiply the sqrt of self.d_model here? see note1
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create the positional embedding tensor
        pe = torch.zeros(seq_len, d_model)

        # the position array (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # the denominator array
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        # If we mutiply position by div_term, then we will trigger the broadcast funciton and we will have a tesor(seq_len, d_model//2)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        # add a dimension for batch
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:(batch_size, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # Here, the pe has been registered, so it is ok to not add the `requires_grad` part. It is not learned by default
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x:torch.Tensor) -> None:
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)

        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        assert d_model % h == 0, "d_model should be divisible by h"
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # (batch_size, h, seq_len, d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.w_q(q) # type: torch.Tensor
        key = self.w_k(k) # type: torch.Tensor
        value = self.w_v(v) # type: torch.Tensor

        # transpose(1, 2) to make sure that each the block is able to see the whole sentence, thought not complete words.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # x is the real output, attention_scores is only for visualization purpose

        # x: (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # in the original paper, they first apply sublayer then norm, but we switch the order here.
        # sublayer: the layer that the ResidualConnection connects
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # We need a src_mask here to avoid the interaction between the [PAD] and normal words
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Here, the lambda expression itself will be passed to forward method of ResidualConnection as a callable.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block # type: MultiHeadAttentionBlock
        self.cross_attention_block = cross_attention_block # type: MultiHeadAttentionBlock
        self.feed_forward_block = feed_forward_block # type: FeedForwardBlock
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)

        x = self.norm(x)
        return x

class ProjectionLayer(nn.Module):
    """
    The last Linear layer
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        return torch.log_softmax(x, dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, encode_embed: InputEmbeddings, decode_embed: InputEmbeddings, pos_embed: PositionalEmbedding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encode_embed = encode_embed
        self.decode_embed = decode_embed
        self.pos_embed = pos_embed
        self.proj = proj

    def encode(self, x: torch.Tensor, encoder_mask: torch.Tensor):
        x = self.encode_embed(x)
        x = self.pos_embed(x)
        return self.encoder(x, encoder_mask)
    
    def decode(self, x: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        x = self.decode_embed(x)
        x = self.pos_embed(x)
        return self.decoder(x, encoder_output, encoder_mask, decoder_mask)
    
    def project(self, x: torch.Tensor):
        return self.proj(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # positional embedding layer
    pos_embed = PositionalEmbedding(d_model, seq_len, dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    proj = ProjectionLayer(d_model, tgt_vocab_size)

    # transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, pos_embed, proj)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer