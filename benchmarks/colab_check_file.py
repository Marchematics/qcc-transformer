from pathlib import Path
p = Path('/content/qcc-transformer/artifacts/colab_none_prefix_result.json')
print('exists', p.exists(), 'size', p.stat().st_size if p.exists() else 0)
if p.exists():
    print(p.read_text())
