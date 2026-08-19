# nnbox 🧰

Reusable PyTorch building blocks and models (attention, transformer blocks, embeddings,
norms, ViT, etc..). These are *not* meant to be SOTA; they're lean implementations meant to unify and accelerate my own research projects. Certain design decisions (eg., RMSNorm, GELU, QK norm) are hardcoded to limit drift.

## Install

Install PyTorch first (with the CUDA build you want), then:

```bash
# development: edits are picked up without reinstalling
pip install -e /path/to/nnbox

# pinned, from another project
pip install "nnbox @ git+https://github.com/samb271/nnbox.git@v0.1.0"
```

## Use

```python
from nnbox import SelfAttention, TransformerBlock
```

`from nnbox.attention import SelfAttention` also works if you prefer the
explicit path.
