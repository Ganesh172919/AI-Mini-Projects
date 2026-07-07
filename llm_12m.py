"""
12M Parameter Language Model - Built from Scratch
===================================================
A decoder-only transformer (GPT-style) with ~12 million trainable parameters.
Designed for learning and experimentation on small-scale text generation tasks.

Architecture:
  - Byte-level tokenization (vocab_size=256)
  - Learned positional embeddings
  - Pre-norm transformer blocks (RMSNorm)
  - Multi-head self-attention with causal masking
  - SwiGLU feed-forward network
  - Weight tying between input embeddings and output projection

Usage:
  python llm_12m.py              # Run demo with synthetic data
  python llm_12m.py --train      # Train on custom text file
  python llm_12m.py --generate   # Generate text from a prompt

Requirements:
  pip install torch numpy
"""

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model and training hyperparameters."""
    # Model architecture (~12M parameters)
    vocab_size: int = 256           # Byte-level tokens
    d_model: int = 256              # Embedding / hidden dimension
    n_layers: int = 12              # Number of transformer blocks
    n_heads: int = 8                # Number of attention heads
    d_ff: int = 1024                # Feed-forward inner dimension
    max_seq_len: int = 512          # Maximum sequence length
    dropout: float = 0.1            # Dropout rate

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 100
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Tokenizer (Byte-Level)
# ============================================================================

class ByteTokenizer:
    """Simple byte-level tokenizer. Each byte (0-255) is a token."""

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


# ============================================================================
# Model Components
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (simpler and faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(xW1) * (xV) -> W2. Better than ReLU FFN."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.v(x))


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute rotary position embeddings (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, dim/2)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Expand cos/sin for batch and heads
    cos = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking and RoPE."""

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

        # RoPE
        cos, sin = precompute_rope(self.d_head, config.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual."""

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ============================================================================
# Full Model
# ============================================================================

class MiniLLM(nn.Module):
    """
    A 12M parameter decoder-only language model.

    Architecture overview:
      Token Embedding -> N x [Attention + FFN] -> RMSNorm -> Output (tied weights)

    Parameter breakdown:
      - Embedding:    256*256 + 512*256 = 196,608
      - 12 layers:    ~1,049,088 each = 12,589,056
      - Final norm:   256
      - Output:       tied with embedding (0 additional)
      - Total:        ~12,785,920 (12.8M)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm and output projection
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: output projection shares weights with token embedding
        self.output.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        device = input_ids.device

        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Embeddings
        positions = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.norm(x)
        logits = self.output(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text autoregressively with temperature, top-k, and top-p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================================
# Dataset
# ============================================================================

class TextDataset(Dataset):
    """Character/byte-level dataset for language model training."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_text_data(path: str, tokenizer: ByteTokenizer) -> torch.Tensor:
    """Load and tokenize a text file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


def create_synthetic_data(tokenizer: ByteTokenizer) -> torch.Tensor:
    """Create synthetic training data for demonstration."""
    patterns = [
        "The quick brown fox jumps over the lazy dog. " * 50,
        "To be or not to be, that is the question. " * 50,
        "In the beginning was the word, and the word was with AI. " * 50,
        "Machine learning is a subset of artificial intelligence. " * 50,
        "Neural networks learn by adjusting weights through backpropagation. " * 50,
        "Deep learning has revolutionized natural language processing. " * 50,
        "Transformers use self-attention mechanisms to process sequences. " * 50,
        "Large language models are trained on massive text corpora. " * 50,
    ]
    text = "\n".join(patterns)
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


# ============================================================================
# Learning Rate Scheduler (Cosine with Warmup)
# ============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step > max_steps:
        return lr * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * (0.1 + 0.9 * coeff)


# ============================================================================
# Training Loop
# ============================================================================

def train(model: MiniLLM, dataset: TextDataset, config: Config):
    """Full training loop with gradient clipping and LR scheduling."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    max_steps = config.max_epochs * len(loader)
    step = 0

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Parameters:   {model.count_parameters():,}")
    print(f"  Device:       {config.device}")
    print(f"  Batch size:   {config.batch_size}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"{'='*60}\n")

    for epoch in range(config.max_epochs):
        total_loss = 0.0
        num_batches = 0

        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)

            # Update learning rate
            lr = get_lr(step, config.warmup_steps, max_steps, config.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            _, loss = model(x, targets=y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            step += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config.max_epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

    print(f"\nTraining complete! Final loss: {avg_loss:.4f}")


# ============================================================================
# Text Generation
# ============================================================================

def generate_text(
    model: MiniLLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    config: Config,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Generate text from a prompt."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    generated = output_ids[0].tolist()
    return tokenizer.decode(generated)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="12M Parameter Language Model")
    parser.add_argument("--train", type=str, help="Path to training text file")
    parser.add_argument("--generate", action="store_true", help="Generate text interactively")
    parser.add_argument("--prompt", type=str, default="The ", help="Generation prompt")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--save", type=str, help="Path to save model checkpoint")
    parser.add_argument("--load", type=str, help="Path to load model checkpoint")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = parser.parse_args()

    # Setup
    config = Config()
    config.max_epochs = args.epochs
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    tokenizer = ByteTokenizer()
    model = MiniLLM(config).to(config.device)

    # Load checkpoint if specified
    if args.load:
        print(f"Loading model from {args.load}...")
        model.load_state_dict(torch.load(args.load, map_location=config.device))

    # Print model info
    print(f"\n{'='*60}")
    print(f"MiniLLM - 12M Parameter Language Model")
    print(f"{'='*60}")
    print(f"Architecture: Decoder-only Transformer")
    print(f"Parameters:   {model.count_parameters():,}")
    print(f"Layers:       {config.n_layers}")
    print(f"Hidden dim:   {config.d_model}")
    print(f"Heads:        {config.n_heads}")
    print(f"FFN dim:      {config.d_ff}")
    print(f"Vocab size:   {config.vocab_size}")
    print(f"Max seq len:  {config.max_seq_len}")
    print(f"Device:       {config.device}")
    print(f"{'='*60}\n")

    # Training mode
    if args.train:
        print(f"Loading training data from {args.train}...")
        data = load_text_data(args.train, tokenizer)
    elif args.demo:
        print("Using synthetic training data for demo...")
        data = create_synthetic_data(tokenizer)
    else:
        data = None

    if data is not None:
        dataset = TextDataset(data, args.seq_len)
        print(f"Dataset size: {len(dataset):,} sequences\n")

        train(model, dataset, config)

        # Save checkpoint
        if args.save:
            torch.save(model.state_dict(), args.save)
            print(f"Model saved to {args.save}")

        # Demo generation after training
        print(f"\n{'='*60}")
        print("Generation Demo")
        print(f"{'='*60}")
        prompts = [
            "The quick brown",
            "Machine learning",
            "Neural networks",
        ]
        for p in prompts:
            text = generate_text(model, tokenizer, p, config, max_tokens=100)
            print(f"\nPrompt: '{p}'")
            print(f"Output: {text[:200]}")

    # Interactive generation mode
    if args.generate and data is None:
        print("Interactive generation mode (type 'quit' to exit)\n")
        while True:
            prompt = input("Prompt: ")
            if prompt.lower() in ("quit", "exit", "q"):
                break
            text = generate_text(model, tokenizer, prompt, config, max_tokens=200)
            print(f"\n{text}\n")

    # Default: run demo if no mode specified
    if data is None and not args.generate:
        print("No mode specified. Running quick demo...\n")
        print("Options:")
        print("  python llm_12m.py --demo              # Train on synthetic data")
        print("  python llm_12m.py --train data.txt     # Train on your text file")
        print("  python llm_12m.py --generate --load model.pt  # Generate from saved model")
        print()

        # Quick demo with synthetic data
        data = create_synthetic_data(tokenizer)
        dataset = TextDataset(data, 128)
        config.max_epochs = 3
        config.batch_size = 16
        train(model, dataset, config)

        print("\nDemo generation:")
        text = generate_text(model, tokenizer, "The quick brown", config, max_tokens=80)
        print(f"Prompt: 'The quick brown'")
        print(f"Output: {text}")


if __name__ == "__main__":
    main()
