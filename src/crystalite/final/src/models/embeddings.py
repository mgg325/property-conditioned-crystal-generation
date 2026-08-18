from __future__ import annotations

import math
import torch
from torch import nn

_DEBUG = False
if "DEBUG" in __import__("os").environ:
    _DEBUG = True

class TimeEmbedder(nn.Module):
    def __init__(self, d_model: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim * 2, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def _timestep_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = 2 * math.pi * t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t_g: torch.Tensor, t_a: torch.Tensor) -> torch.Tensor:
        t_g_emb = self._timestep_embedding(t_g, self.freq_dim)
        t_a_emb = self._timestep_embedding(t_a, self.freq_dim)
        t = torch.cat([t_g_emb, t_a_emb], dim=-1)
        return self.mlp(t)


class ScalarPropertyEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        property_mean: float = 0.0,
        property_std: float = 1.0,
        property_transform: str = "identity",
    ) -> None:
        super().__init__()
        if property_std == 0.0:
            raise ValueError("property_std must be non-zero.")
        self.property_transform = self._normalize_transform(property_transform)
        self.register_buffer(
            "property_mean",
            torch.tensor(float(property_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "property_std",
            torch.tensor(float(property_std), dtype=torch.float32),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.null_embedding = nn.Parameter(torch.zeros(d_model))

    @staticmethod
    def _normalize_transform(property_transform: str) -> str:
        transform = str(property_transform).strip().lower()
        if transform in {"", "none", "raw", "linear", "standardize"}:
            transform = "identity"
        if transform not in {"identity", "log"}:
            raise ValueError(
                "property_transform must be 'identity' or 'log', "
                f"got {property_transform!r}."
            )
        return transform

    def set_normalization(self, property_mean: float, property_std: float) -> None:
        if property_std == 0.0:
            raise ValueError("property_std must be non-zero.")
        with torch.no_grad():
            self.property_mean.fill_(float(property_mean))
            self.property_std.fill_(float(property_std))

    def normalize_property_values(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(
            device=self.property_mean.device,
            dtype=self.property_mean.dtype,
        )
        if self.property_transform == "log":
            values = torch.log(torch.clamp(values, min=1e-6))
        return (values - self.property_mean) / self.property_std

    def forward(
        self,
        property_values: torch.Tensor | None = None,
        force_uncond_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if property_values is None:
            if batch_size is None:
                raise ValueError("batch_size is required when property_values is None.")
            out = self.null_embedding[None, :].expand(int(batch_size), -1)
        else:
            values = property_values
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1:
                raise ValueError(
                    "property_values must have shape (B,) or (B, 1), "
                    f"got {tuple(property_values.shape)}."
                )
            if batch_size is not None and values.shape[0] != int(batch_size):
                raise ValueError(
                    f"property_values batch size {values.shape[0]} does not match "
                    f"batch_size {batch_size}."
                )
            values = self.normalize_property_values(values)[:, None]
            out = self.mlp(values)

        if force_uncond_mask is not None:
            mask = force_uncond_mask.to(device=out.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != out.shape[0]:
                raise ValueError(
                    "force_uncond_mask must have shape (B,), "
                    f"got {tuple(force_uncond_mask.shape)}."
                )
            null = self.null_embedding[None, :].expand_as(out)
            out = torch.where(mask[:, None], null, out)

        if device is not None or dtype is not None:
            out = out.to(device=device if device is not None else out.device, dtype=dtype)
        return out


class SoftClusterScalarPropertyEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int = 32,
        init_min: float = -2.5,
        init_max: float = 4.0,
        property_mean: float = 0.0,
        property_std: float = 1.0,
        property_transform: str = "identity",
    ) -> None:
        super().__init__()
        if int(num_clusters) <= 0:
            raise ValueError(f"num_clusters must be > 0, got {num_clusters}.")
        if property_std == 0.0:
            raise ValueError("property_std must be non-zero.")
        self.hidden_dim = int(hidden_dim)
        self.num_clusters = int(num_clusters)
        self.property_transform = ScalarPropertyEmbedder._normalize_transform(property_transform)
        self.register_buffer(
            "property_mean",
            torch.tensor(float(property_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "property_std",
            torch.tensor(float(property_std), dtype=torch.float32),
        )
        self.centers = nn.Parameter(
            torch.linspace(float(init_min), float(init_max), self.num_clusters)
        )
        self.log_width = nn.Parameter(torch.zeros(self.num_clusters))
        self.mlp = nn.Sequential(
            nn.Linear(self.num_clusters, self.hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        )
        self.null_embedding = nn.Parameter(torch.zeros(self.hidden_dim))

    @staticmethod
    def _normalize_transform(property_transform: str) -> str:
        return ScalarPropertyEmbedder._normalize_transform(property_transform)

    def set_normalization(self, property_mean: float, property_std: float) -> None:
        if property_std == 0.0:
            raise ValueError("property_std must be non-zero.")
        with torch.no_grad():
            self.property_mean.fill_(float(property_mean))
            self.property_std.fill_(float(property_std))

    def normalize_property_values(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(
            device=self.property_mean.device,
            dtype=self.property_mean.dtype,
        )
        if self.property_transform == "log":
            values = torch.log(torch.clamp(values, min=1e-6))
        return (values - self.property_mean) / self.property_std

    def forward(
        self,
        property_values: torch.Tensor | None = None,
        force_uncond_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if property_values is None:
            if batch_size is None:
                raise ValueError("batch_size is required when property_values is None.")
            out = self.null_embedding[None, :].expand(int(batch_size), -1)
        else:
            values = property_values
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1:
                raise ValueError(
                    "property_values must have shape (B,) or (B, 1), "
                    f"got {tuple(property_values.shape)}."
                )
            if batch_size is not None and values.shape[0] != int(batch_size):
                raise ValueError(
                    f"property_values batch size {values.shape[0]} does not match "
                    f"batch_size {batch_size}."
                )
            z = self.normalize_property_values(values)[:, None]
            centers = self.centers.to(device=z.device, dtype=z.dtype)[None, :]
            width = torch.exp(self.log_width.to(device=z.device, dtype=z.dtype))[None, :]
            width = width + torch.finfo(z.dtype).eps
            dist2 = (z - centers) ** 2
            weights = torch.softmax(-dist2 / (width ** 2), dim=-1)
            out = self.mlp(weights)

        if force_uncond_mask is not None:
            mask = force_uncond_mask.to(device=out.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != out.shape[0]:
                raise ValueError(
                    "force_uncond_mask must have shape (B,), "
                    f"got {tuple(force_uncond_mask.shape)}."
                )
            null = self.null_embedding[None, :].expand_as(out)
            out = torch.where(mask[:, None], null, out)

        if device is not None or dtype is not None:
            out = out.to(device=device if device is not None else out.device, dtype=dtype)
        return out


class ScalarPropertyFiLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_film_layers: int,
        property_mean: float = 0.0,
        property_std: float = 1.0,
        property_transform: str = "identity",
    ) -> None:
        super().__init__()
        if n_film_layers <= 0:
            raise ValueError(f"n_film_layers must be > 0, got {n_film_layers}.")
        if property_std == 0.0:
            raise ValueError("property_std must be non-zero.")
        self.d_model = int(d_model)
        self.n_film_layers = int(n_film_layers)
        self.property_transform = ScalarPropertyEmbedder._normalize_transform(property_transform)
        self.register_buffer(
            "property_mean",
            torch.tensor(float(property_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "property_std",
            torch.tensor(float(property_std), dtype=torch.float32),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, 2 * self.n_film_layers * self.d_model, bias=True),
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def set_normalization(self, property_mean: float, property_std: float) -> None:
        if property_std == 0.0:
            raise ValueError("property_std must be non-zero.")
        with torch.no_grad():
            self.property_mean.fill_(float(property_mean))
            self.property_std.fill_(float(property_std))

    def normalize_property_values(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(
            device=self.property_mean.device,
            dtype=self.property_mean.dtype,
        )
        if self.property_transform == "log":
            values = torch.log(torch.clamp(values, min=1e-6))
        return (values - self.property_mean) / self.property_std

    def forward(
        self,
        property_values: torch.Tensor | None = None,
        force_uncond_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if property_values is None:
            if batch_size is None:
                raise ValueError("batch_size is required when property_values is None.")
            out_device = device if device is not None else self.property_mean.device
            out_dtype = dtype if dtype is not None else self.property_mean.dtype
            gamma = torch.zeros(
                (int(batch_size), self.n_film_layers, self.d_model),
                device=out_device,
                dtype=out_dtype,
            )
            beta = torch.zeros_like(gamma)
            return gamma, beta

        values = property_values
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(
                "property_values must have shape (B,) or (B, 1), "
                f"got {tuple(property_values.shape)}."
            )
        if batch_size is not None and values.shape[0] != int(batch_size):
            raise ValueError(
                f"property_values batch size {values.shape[0]} does not match "
                f"batch_size {batch_size}."
            )

        z = self.normalize_property_values(values)[:, None]
        gamma_beta = self.mlp(z).view(values.shape[0], self.n_film_layers, 2, self.d_model)
        gamma = gamma_beta[:, :, 0, :]
        beta = gamma_beta[:, :, 1, :]

        if force_uncond_mask is not None:
            mask = force_uncond_mask.to(device=gamma.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != gamma.shape[0]:
                raise ValueError(
                    "force_uncond_mask must have shape (B,), "
                    f"got {tuple(force_uncond_mask.shape)}."
                )
            gamma = torch.where(mask[:, None, None], torch.zeros_like(gamma), gamma)
            beta = torch.where(mask[:, None, None], torch.zeros_like(beta), beta)

        if device is not None or dtype is not None:
            gamma = gamma.to(device=device if device is not None else gamma.device, dtype=dtype)
            beta = beta.to(device=device if device is not None else beta.device, dtype=dtype)
        return gamma, beta


class FourierCoordEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_freqs: int = 32,
        mode: str = "rff",
        rff_dim: int | None = None,
        rff_sigma: float = 1.0,
    ) -> None:
        """
        Coordinate embedder with switchable deterministic Fourier or Random Fourier Features (RFF).

        Args:
            d_model: output embedding dimension.
            n_freqs: number of deterministic frequencies (when mode="fourier").
            mode: "fourier" (deterministic 1..n_freqs) or "rff" (random Fourier features).
            rff_dim: number of random frequency samples; defaults to n_freqs when None.
            rff_sigma: stddev of the random projection matrix for RFF.
        """
        super().__init__()
        self.mode = mode.lower()

        if self.mode not in {"fourier", "rff"}:
            raise ValueError(f"Unsupported coord embed mode: {mode}")

        if self.mode == "fourier":
            self.n_freqs = n_freqs
            self.register_buffer("freqs", torch.arange(1, n_freqs + 1, dtype=torch.float32))
            in_dim = 6 * n_freqs  # 3 axes, sin+cos
        else:
            self.n_rff = int(rff_dim) if rff_dim is not None else int(n_freqs)
            # Fixed random projection matrix; not trainable but stored as buffer for checkpointing.
            self.register_buffer("proj", torch.randn(3, self.n_rff) * float(rff_sigma))
            in_dim = 2 * self.n_rff  # sin+cos on projected coords

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, frac_coords: torch.Tensor) -> torch.Tensor:
        # frac_coords: (B, N, 3), expected in [0,1)
        if self.mode == "fourier":
            args = 2 * math.pi * frac_coords[..., None] * self.freqs[None, None, None, :]
            sin = torch.sin(args)
            cos = torch.cos(args)
            feats = torch.cat([sin, cos], dim=-1)
            feats = feats.reshape(frac_coords.shape[0], frac_coords.shape[1], -1)
        else:
            # Project coords with fixed random matrix then apply sinusoidal RFF.
            proj = frac_coords @ self.proj  # (B, N, n_rff)
            args = 2 * math.pi * proj
            feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(feats)


class LatticeEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        mode: str = "rff",
        rff_dim: int = 256,
        rff_sigma: float = 5.0,
    ) -> None:
        """
        Lattice embedder with switchable modes:
          - "mlp": legacy two-layer MLP over raw lattice features (log lengths + cos angles)
          - "rff": random Fourier features with fixed Gaussian projection, followed by MLP
        """
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"mlp", "rff"}:
            raise ValueError(f"Unsupported lattice embed mode: {mode}")

        if self.mode == "mlp":
            in_dim = 6
        else:
            self.n_rff = int(rff_dim)
            self.register_buffer("proj", torch.randn(6, self.n_rff) * float(rff_sigma))
            in_dim = 2 * self.n_rff  # sin + cos

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp":
            feats = y
        else:
            proj = y @ self.proj  # (B, n_rff)
            args = 2 * math.pi * proj
            feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(feats)


class MaskEmbedder(nn.Module):
    def __init__(self, d_model: int, in_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(mask.float())


class TokenEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vz: int,
        n_freqs: int = 32,
        coord_embed_mode: str = "rff",
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        lattice_embed_mode: str = "rff",
        lattice_rff_dim: int = 256,
        lattice_rff_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        # PAD=0, elements=1..vz, MASK=vz+1
        self.type_embed = nn.Embedding(vz + 2, d_model, padding_idx=0)
        self.coord_embed = FourierCoordEmbedder(
            d_model=d_model,
            n_freqs=n_freqs,
            mode=coord_embed_mode,
            rff_dim=coord_rff_dim,
            rff_sigma=coord_rff_sigma,
        )
        self.lattice_embed = LatticeEmbedder(
            d_model=d_model,
            mode=lattice_embed_mode,
            rff_dim=lattice_rff_dim,
            rff_sigma=lattice_rff_sigma,
        )
        self.mask_embed_atom = MaskEmbedder(d_model=d_model, in_dim=2)
        self.mask_embed_lat = MaskEmbedder(d_model=d_model, in_dim=1)
        self.segment_embed = nn.Embedding(2, d_model)

    def forward(
        self,
        a_t: torch.Tensor,
        f_t: torch.Tensor,
        y_t: torch.Tensor,
        m_a: torch.Tensor,
        m_f: torch.Tensor,
        m_y: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Atom tokens
        h_a = self.type_embed(a_t) + self.coord_embed(f_t)
        mask_atom = torch.stack([m_a, m_f], dim=-1)
        h_a = h_a + self.mask_embed_atom(mask_atom) + self.segment_embed.weight[0]

        # Lattice token
        h_y = self.lattice_embed(y_t) + self.mask_embed_lat(m_y) + self.segment_embed.weight[1]
        h_y = h_y[:, None, :]

        # Sequence concat and pad mask (lattice token is always valid)
        x = torch.cat([h_a, h_y], dim=1)
        pad_mask_seq = torch.cat(
            [pad_mask.bool(), torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool)],
            dim=1,
        )
        if _DEBUG:
            if pad_mask_seq.shape[1] != a_t.shape[1] + 1:
                raise RuntimeError(
                    f"pad_mask_seq shape {tuple(pad_mask_seq.shape)} does not match a_t {tuple(a_t.shape)}"
                )
            if pad_mask_seq[:, -1].any():
                raise RuntimeError("Lattice token should never be masked.")
        return x, pad_mask_seq
