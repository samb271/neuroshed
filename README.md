# neuroshed 🧰

Reusable PyTorch building blocks and models (attention, transformer blocks, embeddings,
norms, ViT, etc..). These are *not* meant to be SOTA; they're lean implementations meant to unify and accelerate my own research projects. Certain design decisions (eg., RMSNorm, GELU, QK norm) are hardcoded to limit drift.

## Install

Install PyTorch first (with the CUDA build you want), then:

```bash
# development: edits are picked up without reinstalling
pip install -e /path/to/neuroshed

# pinned, from another project
pip install "neuroshed @ git+https://github.com/samb271/neuroshed.git@v0.1.0"
```

## Use

```python
from neuroshed import SelfAttention, TransformerBlock
```

`from neuroshed.attention import SelfAttention` also works if you prefer the
explicit path.

---

## TODO

- [ ] SIGReg
- [ ] EMA
- [ ] VICReg
- [ ] CNN encoder
- [ ] Embedding layer