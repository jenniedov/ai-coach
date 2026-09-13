"""Speed patch for mlx-audio's Breeze TTS 2 depth decoder.

The stock loop samples each of the 15 sub-codebooks with `int(token.item())`, which forces a GPU
sync 15 times per audio frame (~190 syncs per second of audio). This version keeps the tokens on
device and syncs once per frame. Same maths, same sampler, same masking; only the scheduling changes."""
from __future__ import annotations

import types

import mlx.core as mx
import mlx.nn as nn
from mlx_audio.lm.sample_utils import make_sampler


def _fast_depth_tokens(self, first_codebook, conditional_hidden, *, unconditional_hidden, cfg_scale,
                       temperature, top_p, top_k):
    valid = self.vocab_size
    effective_top_k = min(top_k, valid) if top_k else 0
    if effective_top_k == valid:
        effective_top_k = 0
    sampler = make_sampler(temp=temperature, top_p=top_p, top_k=effective_top_k)
    tokens = mx.array([[0, int(first_codebook)]], dtype=mx.int32)
    for _ in range(self.num_codebooks - 1):
        logits = self.depth_decoder.next_logits(tokens, conditional_hidden)
        if unconditional_hidden is not None:
            unconditional_logits = self.depth_decoder.next_logits(tokens, unconditional_hidden)
            logits = unconditional_logits + cfg_scale * (logits - unconditional_logits)
        logits = self._mask_reserved_codec_logits(logits)[..., :valid]
        token = sampler(nn.log_softmax(logits, axis=-1)).astype(mx.int32).reshape(1, 1)
        tokens = mx.concatenate([tokens, token], axis=1)
    mx.eval(tokens)
    return tokens[0, 1:].tolist()


def apply(model) -> None:
    model._depth_tokens = types.MethodType(_fast_depth_tokens, model)
