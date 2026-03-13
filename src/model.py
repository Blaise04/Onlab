"""
Neurális háló architektúrák Black-Scholes opciós árazáshoz.

Implementált modellek:
  - MLPPricer         : Culkin & Das (2017) baseline MLP
  - DeepMLPPricer     : Lürig et al. (2023) javított MLP (LayerNorm + Dropout)
  - ResNetPricer      : Lürig et al. (2023) reziduális MLP
  - GELUResNetPricer  : ResNetPricer GELU aktivációval (baseline összehasonlítás)
  - DenseMLPPricer    : DenseNet-stílusú MLP (Huang et al. 2017)
  - HighwayPricer     : Highway Network tanulható gating-gel
  - FINNPricer        : Finance-Informed NN — BS közelítő + korrekciós ág
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# GELUResNetPricer — ResNetPricer GELU aktivációval
# ---------------------------------------------------------------------------

class _GELUResidualBlock(nn.Module):
    """Pre-LN reziduális blokk GELU-val: x → LayerNorm → Linear → GELU → Dropout → Linear → + x"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GELUResNetPricer(nn.Module):
    """
    ResNetPricer GELU aktivációval — baseline összehasonlításhoz.

    A BS árak simák, a GELU simább gradienst biztosít (nincs "törött" derivált
    0-nál, szemben a ReLU-val).

    Architektúra (azonos ResNetPricer-rel, nn.ReLU() → nn.GELU()):
        Input(5) → Linear(5→256) → GELU       ← input projekció
                 → [GELUResidualBlock(256)] × n_blocks
                 → LayerNorm → Linear(256→1)

    Paraméterek (default): ~400 000
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_GELUResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# DenseMLPPricer — DenseNet-stílusú összefűzéses MLP
# ---------------------------------------------------------------------------

class DenseMLPPricer(nn.Module):
    """
    DenseNet-stílusú MLP — minden réteg az összes korábbi kimenetét kapja inputként.

    Architektúra (Huang et al. 2017 alapján):
        h₁ = GELU(W₁·x)
        h₂ = GELU(W₂·[x, h₁])
        h₃ = GELU(W₃·[x, h₁, h₂])
        h₄ = GELU(W₄·[x, h₁, h₂, h₃])
        output = W_out·[x, h₁, h₂, h₃, h₄]

    Előnyök: jobb gradiens-áramlás, korai rétegek direkt kapcsolódnak a kimenethez.

    Paraméterek (default, input_dim=5, hidden_dim=128, n_layers=4): ~102 000
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        in_dim = input_dim
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim += hidden_dim  # következő réteg: összes korábbi kimenet + input

        # Kimeneti vetítés: input + összes rejtett réteg
        final_dim = input_dim + n_layers * hidden_dim
        self.output_proj = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for layer in self.layers:
            h_in = torch.cat(outputs, dim=-1)
            h = self.dropout(F.gelu(layer(h_in)))
            outputs.append(h)
        return self.output_proj(torch.cat(outputs, dim=-1))


# ---------------------------------------------------------------------------
# HighwayPricer — tanulható gating
# ---------------------------------------------------------------------------

class _HighwayBlock(nn.Module):
    """
    Highway blokk: H·T + x·(1−T)
        H = GELU(W_H·x + b_H)   — transform
        T = σ(W_T·x + b_T)      — transform gate
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Gate bias negatívra inicializálva → kezdetben inkább "carry" (skip)
        nn.init.constant_(self.T.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.dropout(F.gelu(self.H(x)))
        T = torch.sigmoid(self.T(x))
        return H * T + x * (1.0 - T)


class HighwayPricer(nn.Module):
    """
    Highway Network tanulható gating-gel.

    A skip arány nem rögzített (mint ResNetben), hanem tanult: a háló maga
    dönti el, mikor "enged át" és mikor "transzformál" (Srivastava et al. 2015).

    Architektúra:
        Input(5) → Linear(5→256) → GELU
                 → [HighwayBlock(256)] × n_blocks
                 → Linear(256→1)

    Paraméterek (default): ~528 000
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256,
                 n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_HighwayBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# FINNPricer — Finance-Informed Neural Network
# ---------------------------------------------------------------------------

class FINNPricer(nn.Module):
    """
    Finance-Informed NN — két-ágú architektúra, Liu et al. (2019) / FINN stílusban.

    Inspiráció: arXiv:2412.12213 (AI Black-Scholes)

    Architektúra:
        Ág 1 (approx):    x → kis MLP (2 réteg, 64 neuron) → BS̃
        Ág 2 (correction): x → Linear → GELU → [GELUResidualBlock] × n_blocks
                             → LayerNorm → Linear → δ
        Output: BS̃ + δ

    Az approx ág a "könnyű" eseteket közelíti, a correction ág a nehéz
    eseteket (mélyen OTM, rövid lejárat) korrigálja.

    Paraméterek (default): ~402 000
    """

    def __init__(self, input_dim: int = 5, approx_dim: int = 64,
                 resnet_dim: int = 256, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()

        # Ág 1: kis MLP — BS közelítés
        self.approx = nn.Sequential(
            nn.Linear(input_dim, approx_dim),
            nn.GELU(),
            nn.Linear(approx_dim, approx_dim),
            nn.GELU(),
            nn.Linear(approx_dim, 1),
        )

        # Ág 2: reziduális ResNet — korrekció
        self.correction_proj = nn.Sequential(
            nn.Linear(input_dim, resnet_dim),
            nn.GELU(),
        )
        self.correction_blocks = nn.Sequential(
            *[_GELUResidualBlock(resnet_dim, dropout) for _ in range(n_blocks)]
        )
        self.correction_norm = nn.LayerNorm(resnet_dim)
        self.correction_out = nn.Linear(resnet_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs_approx = self.approx(x)
        delta = self.correction_proj(x)
        delta = self.correction_blocks(delta)
        delta = self.correction_norm(delta)
        delta = self.correction_out(delta)
        return bs_approx + delta


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
        name: 'mlp' | 'deep_mlp' | 'resnet' | 'gelu_resnet' | 'dense_mlp'
              | 'highway' | 'finn'
        **kwargs: az adott modell __init__ argumentumai

    Returns:
        Inicializált nn.Module példány
    """
    registry = {
        'mlp':          MLPPricer,
        'deep_mlp':     DeepMLPPricer,
        'resnet':       ResNetPricer,
        'gelu_resnet':  GELUResNetPricer,
        'dense_mlp':    DenseMLPPricer,
        'highway':      HighwayPricer,
        'finn':         FINNPricer,
    }
    if name not in registry:
        raise ValueError(f"Ismeretlen modell: '{name}'. Válasszon: {list(registry)}")
    return registry[name](**kwargs)
