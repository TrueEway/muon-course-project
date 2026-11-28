"""Record installed package versions without credentials or private hosts."""
import importlib.metadata
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
packages = {}
for name in ['torch','numpy','matplotlib','Pillow','pypdf']:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
target = root/'output'/'revision'/'software_versions.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(packages,indent=2)+'\n',encoding='utf-8')
print(target)
