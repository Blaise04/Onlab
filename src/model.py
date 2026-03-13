"""
Neurális háló architektúrák Black-Scholes opciós árazáshoz.

Implementált modellek:
  - MLPPricer      : Culkin & Das (2017) baseline MLP
  - DeepMLPPricer  : Lürig et al. (2023) javított MLP (LayerNorm + Dropout)
  - ResNetPricer   : Lürig et al. (2023) reziduális MLP
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLPPricer — Culkin & Das (2017) baseline
# ---------------------------------------------------------------------------

class MLPPricer(nn.Module):
    """
    Egyszerű többrétegű perceptron, Culkin & Das (2017) alapján.

    Architektúra:
        Input(5) → Linear(5→100) → ReLU
                 → [Linear(100→100) → ReLU] × (n_layers - 1)
                 → Linear(100→1)

    Paraméterek (default): ~30 900
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 100, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DeepMLPPricer — Lürig et al. (2023) javított MLP
# ---------------------------------------------------------------------------

class DeepMLPPricer(nn.Module):
    """
    Mélyebb MLP LayerNorm-mal és Dropout-tal, Lürig et al. (2023) alapján.

    Architektúra (Pre-LN stílus):
        Input(5) → Linear(5→256)
                 → [LayerNorm → ReLU → Dropout → Linear(256→256)] × n_layers
                 → LayerNorm → Linear(256→1)

    Paraméterek (default): ~265 000
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.Sequential(*blocks)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# ResNetPricer — Lürig et al. (2023) reziduális MLP
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Pre-LN reziduális blokk: x → LayerNorm → Linear → ReLU → Dropout → Linear → + x"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetPricer(nn.Module):
    """
    Reziduális MLP Pre-LN blokkokkal, Lürig et al. (2023) alapján.

    Architektúra:
        Input(5) → Linear(5→256) → ReLU        ← input projekció
                 → [ResidualBlock(256)] × n_blocks
                 → LayerNorm → Linear(256→1)

    Paraméterek (default): ~400 000
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Visszaadja a tanítható paraméterek számát."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Modell factory függvény.

    Args:
        name: 'mlp' | 'deep_mlp' | 'resnet'
        **kwargs: az adott modell __init__ argumentumai

    Returns:
        Inicializált nn.Module példány
    """
    registry = {
        'mlp':      MLPPricer,
        'deep_mlp': DeepMLPPricer,
        'resnet':   ResNetPricer,
    }
    if name not in registry:
        raise ValueError(f"Ismeretlen modell: '{name}'. Válasszon: {list(registry)}")
    return registry[name](**kwargs)
