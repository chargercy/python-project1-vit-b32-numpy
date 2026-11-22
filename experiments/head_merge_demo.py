"""Compare two ways of merging attention heads, to document the shape bug.

Run from the project directory:

    python experiments/head_merge_demo.py

The wrong variant merges [B,H,N,64] with a direct reshape, which interleaves
token rows and heads. The correct variant first moves the head axis back next
to the token axis (transpose 0,2,1,3) and only then flattens the feature axis.
"""
import sys
from pathlib import Path
import numpy as np

# Allow running this file from any directory: the encoder package lives one
# level above this experiments folder.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from runtime import ROOT, load_weights, load_images
from image_encoder import ImageEncoder
from helpers import attention_weights


def attention_variant(attention, x, merge_mode):
    """Re-run one attention sub-layer with the requested head-merge order."""
    qkv = x @ attention.qkv_weight.T + attention.qkv_bias
    width = attention.width
    q = attention.split_heads(qkv[:, :, :width])
    k = attention.split_heads(qkv[:, :, width:2 * width])
    v = attention.split_heads(qkv[:, :, 2 * width:])

    scores = (q @ k.swapaxes(-1, -2)) * np.float32(1.0 / np.sqrt(attention.head_dim))
    coefficients = attention_weights(scores)
    context = coefficients @ v  # [B,H,N,64]

    if merge_mode == 'wrong':
        merged = context.reshape(x.shape[0], x.shape[1], width)
    else:
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], width)
    return merged @ attention.out_weight.T + attention.out_bias


def run_block(encoder, images, merge_mode):
    """Full block 0 forward pass with the chosen merge order."""
    block = encoder.blocks[0]
    x = encoder.pre_norm(encoder.embed(images))
    update = attention_variant(block.attention, block.norm1(x), merge_mode)
    residual = x + update
    u = block.norm2(residual)
    hidden = u @ block.fc1_weight.T + block.fc1_bias
    activated = block.gelu(hidden)
    mlp_update = activated @ block.fc2_weight.T + block.fc2_bias
    return residual + mlp_update


def main():
    encoder = ImageEncoder(load_weights())
    images = load_images()[:2]
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
        expected = ref['block0_tokens'][:2]
    print('block0 reference shape:', expected.shape)
    for mode in ('wrong', 'correct'):
        actual = run_block(encoder, images, mode)
        error = np.abs(actual - expected)
        print(f'{mode:8} merge: max_abs={error.max():.3e} mean_abs={error.mean():.3e}')
        if mode == 'wrong':
            merged_wrong = attention_variant(encoder.blocks[0].attention,
                                             encoder.pre_norm(encoder.embed(images))[:1], 'wrong')
            merged_right = attention_variant(encoder.blocks[0].attention,
                                             encoder.pre_norm(encoder.embed(images))[:1], 'correct')
            print('head-merge difference on the attention update:',
                  f'max_abs={np.abs(merged_wrong - merged_right).max():.3e}',
                  'shape', merged_wrong.shape)


if __name__ == '__main__':
    main()
