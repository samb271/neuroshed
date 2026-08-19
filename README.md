# ml_modules

Reusable PyTorch building blocks (attention, transformer blocks, embeddings,
norms) shared across projects.

## Install

Install PyTorch first (with the CUDA build you want), then:

```bash
# development: edits are picked up without reinstalling
pip install -e /path/to/ml_modules

# pinned, from another project
pip install "ml-modules @ git+https://github.com/samb271/ml_modules.git@v0.1.0"
```

## Use

```python
from ml_modules import SelfAttention, TransformerBlock
```

`from ml_modules.attention import SelfAttention` also works if you prefer the
explicit path.
