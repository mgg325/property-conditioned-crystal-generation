from __future__ import annotations

import torch
from torch import nn

from src.models.embeddings import (
    FourierCoordEmbedder,
    LatticeEmbedder,
    ScalarPropertyFiLM,
    ScalarPropertyEmbedder,
    SoftClusterScalarPropertyEmbedder,
    TimeEmbedder,
)
from src.models.transformer import TransformerTrunk
from src.models.heads import CrystalHeads


def mod1(x: torch.Tensor) -> torch.Tensor:
    """Wrap values into [0, 1)."""
    return x - torch.floor(x)


class CrystaliteModel(nn.Module):
    """
    Transformer backbone for Crystalite EDM training on tokenized crystals.

    This keeps the original trunk/head architecture but swaps the token
    embedding to accept continuous (noisy) features:
      - type_feats: (B, N, type_dim) relaxed type features
      - frac_coords: (B, N, 3) fractional coords (wrapped to [0,1) internally)
      - lattice_feats: (B, 6)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vz: int,
        type_dim: int | None = None,
        n_freqs: int = 32,
        coord_embed_mode: str = "rff",
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        lattice_embed_mode: str = "rff",
        lattice_rff_dim: int = 256,
        lattice_rff_sigma: float = 5.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_distance_bias: bool = False,
        use_edge_bias: bool = False,
        edge_bias_n_freqs: int = 8,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        pbc_radius: int = 1,
        lattice_repr: str = "y1",
        dist_slope_init: float = -1.0,
        use_noise_gate: bool = True,
        gem_per_layer: bool = False,
        coord_head_mode: str = "direct",
        use_property_conditioning: bool = False,
        property_encoder_type: str = "scalar_mlp",
        property_fusion_mode: str = "add",
        property_num_clusters: int = 32,
        property_cluster_min: float = -2.5,
        property_cluster_max: float = 4.0,
        property_mean: float = 0.0,
        property_std: float = 1.0,
        property_transform: str = "identity",
        use_property_film: bool = False,
        film_start_layer: int | None = None,
        film_num_layers: int | None = None,
    ) -> None:
        super().__init__()
        if film_start_layer is None or int(film_start_layer) < 0:
            resolved_film_start_layer = int(n_layers) // 2
        else:
            resolved_film_start_layer = int(film_start_layer)
        if film_num_layers is None or int(film_num_layers) < 0:
            resolved_film_num_layers = int(n_layers) - resolved_film_start_layer
        else:
            resolved_film_num_layers = int(film_num_layers)
        if use_property_film:
            if resolved_film_start_layer < 0 or resolved_film_start_layer >= int(n_layers):
                raise ValueError(
                    f"film_start_layer must be in [0, {int(n_layers) - 1}], "
                    f"got {resolved_film_start_layer}."
                )
            if resolved_film_num_layers != int(n_layers) - resolved_film_start_layer:
                raise ValueError(
                    "film_num_layers must cover the selected upper-layer suffix. "
                    f"Expected {int(n_layers) - resolved_film_start_layer}, "
                    f"got {resolved_film_num_layers}."
                )
        self.type_dim = (vz + 1) if type_dim is None else int(type_dim)
        self.use_property_film = bool(use_property_film)
        self.film_start_layer = resolved_film_start_layer
        self.film_num_layers = resolved_film_num_layers
        self.property_conditioning_enabled = bool(use_property_conditioning)
        self.property_encoder_type = str(property_encoder_type).strip().lower()
        self.property_fusion_mode = str(property_fusion_mode).strip().lower()
        if self.property_encoder_type not in {"scalar_mlp", "soft_cluster"}:
            raise ValueError(
                "property_encoder_type must be 'scalar_mlp' or 'soft_cluster', "
                f"got {property_encoder_type!r}."
            )
        if self.property_fusion_mode not in {"add", "gated_add", "concat_mlp", "residual_delta", "none"}:
            raise ValueError(
                "property_fusion_mode must be 'add', 'gated_add', 'concat_mlp', 'residual_delta', or 'none', "
                f"got {property_fusion_mode!r}."
            )
        self.type_proj = nn.Sequential(
            nn.Linear(self.type_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
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
        self.segment_embed = nn.Embedding(2, d_model)
        self.time = TimeEmbedder(d_model=d_model)
        if self.property_conditioning_enabled and self.property_fusion_mode != "none":
            if self.property_encoder_type == "scalar_mlp":
                self.property_embed = ScalarPropertyEmbedder(
                    d_model=d_model,
                    property_mean=property_mean,
                    property_std=property_std,
                    property_transform=property_transform,
                )
            elif self.property_encoder_type == "soft_cluster":
                self.property_embed = SoftClusterScalarPropertyEmbedder(
                    hidden_dim=d_model,
                    num_clusters=property_num_clusters,
                    init_min=property_cluster_min,
                    init_max=property_cluster_max,
                    property_mean=property_mean,
                    property_std=property_std,
                    property_transform=property_transform,
                )
            else:
                raise AssertionError("unreachable property encoder type")
        else:
            self.property_embed = None
        self.property_alpha = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            if self.property_conditioning_enabled and self.property_fusion_mode == "gated_add"
            else None
        )
        self.property_fuse = (
            nn.Sequential(
                nn.Linear(2 * d_model, d_model, bias=True),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
            )
            if self.property_conditioning_enabled and self.property_fusion_mode == "concat_mlp"
            else None
        )
        self.property_delta = (
            nn.Sequential(
                nn.Linear(2 * d_model, 4 * d_model, bias=True),
                nn.SiLU(),
                nn.Linear(4 * d_model, d_model, bias=True),
            )
            if self.property_conditioning_enabled and self.property_fusion_mode == "residual_delta"
            else None
        )
        if self.property_delta is not None:
            nn.init.zeros_(self.property_delta[-1].weight)
            nn.init.zeros_(self.property_delta[-1].bias)
        self.property_film = (
            ScalarPropertyFiLM(
                d_model=d_model,
                n_film_layers=resolved_film_num_layers,
                property_mean=property_mean,
                property_std=property_std,
                property_transform=property_transform,
            )
            if use_property_film
            else None
        )
        self.trunk = TransformerTrunk(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_distance_bias=use_distance_bias,
            use_edge_bias=use_edge_bias,
            edge_bias_n_freqs=edge_bias_n_freqs,
            edge_bias_hidden_dim=edge_bias_hidden_dim,
            edge_bias_n_rbf=edge_bias_n_rbf,
            edge_bias_rbf_max=edge_bias_rbf_max,
            pbc_radius=pbc_radius,
            lattice_repr=lattice_repr,
            dist_slope_init=dist_slope_init,
            use_noise_gate=use_noise_gate,
            gem_per_layer=gem_per_layer,
        )
        self.heads = CrystalHeads(
            d_model=d_model,
            vz=vz,
            type_out_dim=self.type_dim,
            coord_head_mode=coord_head_mode,
        )

    def forward(
        self,
        type_feats: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        t_sigma: torch.Tensor,
        lattice_bias_feats: torch.Tensor | None = None,
        property_values: torch.Tensor | None = None,
        force_uncond_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            type_feats: (B, N, type_dim) relaxed type features with padding zeroed.
            frac_coords: (B, N, 3) fractional coords (not necessarily wrapped).
            lattice_feats: (B, 6)
            pad_mask: (B, N) bool where True denotes padding.
            t_sigma: (B,) scalar noise embedding; shared for t_g/t_a.
            lattice_bias_feats: optional (B, 6) lattice features used only for
                geometry-aware attention bias. If None, lattice_feats is used.
            property_values: optional (B,) or (B, 1) scalar property values.
            force_uncond_mask: optional (B,) bool mask selecting null property
                conditioning for classifier-free guidance training/sampling.
        """
        frac_mod = mod1(frac_coords)
        h_type = self.type_proj(type_feats) + self.coord_embed(frac_mod) + self.segment_embed.weight[0]
        h_lat = self.lattice_embed(lattice_feats) + self.segment_embed.weight[1]
        h_lat = h_lat[:, None, :]

        x = torch.cat([h_type, h_lat], dim=1)
        pad_seq = torch.cat(
            [
                pad_mask.bool(),
                torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
        t_emb_sigma = self.time(t_sigma, t_sigma)
        t_emb = t_emb_sigma
        if self.property_embed is not None and property_values is not None:
            prop_emb = self.property_embed(
                property_values=property_values,
                force_uncond_mask=force_uncond_mask,
                batch_size=type_feats.shape[0],
                device=type_feats.device,
                dtype=t_emb_sigma.dtype,
            )
            if self.property_fusion_mode == "add":
                t_emb = t_emb_sigma + prop_emb
            elif self.property_fusion_mode == "gated_add":
                if self.property_alpha is None:
                    raise RuntimeError("property_alpha is missing for gated_add fusion.")
                t_emb = t_emb_sigma + self.property_alpha.to(dtype=prop_emb.dtype) * prop_emb
            elif self.property_fusion_mode == "concat_mlp":
                if self.property_fuse is None:
                    raise RuntimeError("property_fuse is missing for concat_mlp fusion.")
                t_emb = self.property_fuse(torch.cat([t_emb_sigma, prop_emb], dim=-1))
            elif self.property_fusion_mode == "residual_delta":
                if self.property_delta is None:
                    raise RuntimeError("property_delta is missing for residual_delta fusion.")
                delta = self.property_delta(torch.cat([t_emb_sigma, prop_emb], dim=-1))
                t_emb = t_emb_sigma + delta
            else:
                raise ValueError(f"Unknown property_fusion_mode: {self.property_fusion_mode}")
        property_film = None
        if self.property_film is not None:
            property_film = self.property_film(
                property_values=property_values,
                force_uncond_mask=force_uncond_mask,
                batch_size=type_feats.shape[0],
                device=type_feats.device,
                dtype=t_emb.dtype,
            )
        # In the EDM codepath, t_sigma is the noise embedding c_noise = 0.25 * log(sigma).
        # GEM's noise gate expects sigma (positive, on the same scale as the Karras schedule),
        # so convert back here while keeping the time embedding on c_noise.
        sigma_for_gem = torch.exp(4.0 * t_sigma.to(dtype=torch.float32))
        lattice_for_bias = lattice_feats if lattice_bias_feats is None else lattice_bias_feats
        h = self.trunk(
            x,
            t_emb,
            pad_mask=pad_seq,
            coords=frac_mod,
            lattice=lattice_for_bias,
            t_sigma=sigma_for_gem,
            property_film=property_film,
            film_start_layer=self.film_start_layer if property_film is not None else None,
        )
        return self.heads(h, frac_coords=frac_mod, pad_mask=pad_mask.bool())


# Backwards-compatible alias during the rename from the old MP20-specific name.
MP20EDMModel = CrystaliteModel
