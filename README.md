# neuroshed

Reusable PyTorch building blocks (attention, transformer blocks, embeddings,
norms) shared across projects.

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
