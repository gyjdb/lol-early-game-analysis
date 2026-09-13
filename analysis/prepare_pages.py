"""Stage only public site files, excluding the raw source and development files."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / '.site'
out.mkdir(exist_ok=True)
shutil.copy2(root / 'index.html', out / 'index.html')
shutil.copytree(root / 'assets', out / 'assets', dirs_exist_ok=True)
(out / 'analysis').mkdir(exist_ok=True)
for pattern in ['*.csv', '*.json', 'METHODOLOGY.md']:
    for file in (root / 'analysis').glob(pattern):
        shutil.copy2(file, out / 'analysis' / file.name)
(out / '.nojekyll').touch()
