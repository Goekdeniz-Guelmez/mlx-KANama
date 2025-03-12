from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .kan import KANLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llama"
    hidden_size: int = 3072
    num_hidden_layers: int = 28
    intermediate_size: int = 8192
    num_attention_heads: int = 24
    rms_norm_eps: float = 1e-05
    vocab_size: int = 128256
    head_dim: Optional[int] = 128
    max_position_embeddings: Optional[int] = 131072
    num_key_value_heads: Optional[int] = 8
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = field(
        default_factory=lambda: {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    )
    tie_word_embeddings: bool = True
    
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    hidden_act: Any = nn.SiLU
    grid_eps: float = 0.02
    grid_range: List[float] = field(default_factory=lambda: [-1, 1])


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale = head_dim**-0.5
        attention_bias = getattr(args, "attention_bias", False)
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = KANLinear(
            in_features=args.hidden_size,
            out_features=args.intermediate_size,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            scale_noise=args.scale_noise,
            scale_base=args.scale_base,
            scale_spline=args.scale_spline,
            hidden_act=args.hidden_act,
            grid_eps=args.grid_eps,
            grid_range=args.grid_range,
            bias=args.mlp_bias
        )
        self.gate_proj = KANLinear(
            in_features=args.hidden_size,
            out_features=args.intermediate_size,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            scale_noise=args.scale_noise,
            scale_base=args.scale_base,
            scale_spline=args.scale_spline,
            hidden_act=args.hidden_act,
            grid_eps=args.grid_eps,
            grid_range=args.grid_range,
            bias=args.mlp_bias
        )
        self.down_proj = KANLinear(
            in_features=args.intermediate_size,
            out_features=args.hidden_size,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            scale_noise=args.scale_noise,
            scale_base=args.scale_base,
            scale_spline=args.scale_spline,
            hidden_act=None,
            grid_eps=args.grid_eps,
            grid_range=args.grid_range,
            bias=args.mlp_bias
        )

    def __call__(self, x, update_grid: bool = False) -> mx.array:
        # Get the input shape
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape if needed to ensure correct dimensions
        if hidden_dim != self.gate_proj.in_features:
            # This is a critical error - the dimensions should match by design
            raise ValueError(f"Input dimension {hidden_dim} doesn't match expected {self.gate_proj.in_features}")
        
        # For sequence inputs, we might need to reshape
        if len(x.shape) > 2:
            # Reshape to (batch_size * seq_len, hidden_dim)
            x_reshaped = x.reshape(-1, hidden_dim)
        else:
            x_reshaped = x
            
        if update_grid:
            self.up_proj.update_grid(x_reshaped)
            self.gate_proj.update_grid(x_reshaped)
            
        gate_output = self.gate_proj(x_reshaped)
        up_output = self.up_proj(x_reshaped)
        
        intermediate = gate_output * up_output
        
        if update_grid:
            self.down_proj.update_grid(intermediate)
            
        output = self.down_proj(intermediate)
        
        # Reshape back to original sequence format if needed
        if len(x.shape) > 2:
            output = output.reshape(batch_size, seq_len, -1)
            
        return output
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Calculate the regularization loss for all KAN layers in this MLP"""
        up_loss = self.up_proj.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy
        )
        gate_loss = self.gate_proj.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy
        )
        down_loss = self.down_proj.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy
        )
        return up_loss + gate_loss + down_loss


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        update_grid: bool = False,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        normalized_h = self.post_attention_layernorm(h)
        r = self.mlp(normalized_h, update_grid=update_grid)
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        update_grid: bool = False,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c, update_grid=update_grid)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        update_grid: bool = False,
    ):
        out = self.model(inputs, mask, cache, update_grid=update_grid)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Calculate the regularization loss for all KAN layers in the model"""
        total_loss = 0.0
        for layer in self.layers:
            if hasattr(layer.mlp, 'regularization_loss'):
                total_loss += layer.mlp.regularization_loss(
                    regularize_activation=regularize_activation,
                    regularize_entropy=regularize_entropy
                )
        return total_loss

    def sanitize(self, weights):
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers