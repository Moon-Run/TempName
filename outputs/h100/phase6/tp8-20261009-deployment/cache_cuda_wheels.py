"""Repackage installed Python-independent CUDA wheels after RECORD verification."""
import base64
import csv
import hashlib
from importlib.metadata import distributions
from pathlib import Path
import zipfile

out = Path(__file__).parent / 'wheelhouse'
out.mkdir(exist_ok=True)
for dist in distributions():
    name = dist.metadata['Name']
    if not name.startswith('nvidia-'):
        continue
    tags = [line.split(': ', 1)[1] for line in dist.read_text('WHEEL').splitlines()
            if line.startswith('Tag: ')]
    assert tags and all(t.startswith('py3-none-') for t in tags), (name, tags)
    root = Path(dist.locate_file('')).resolve()
    rows = list(csv.reader(dist.read_text('RECORD').splitlines()))
    target = out / f"{name.replace('-', '_')}-{dist.version}-{tags[0]}.whl"
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as wheel:
        for relative, digest, size in rows:
            path = root / relative
            assert path.resolve().is_relative_to(root), relative
            if '__pycache__' in path.parts or relative.endswith('.pyc'):
                continue
            if digest:
                algorithm, expected = digest.split('=', 1)
                assert algorithm == 'sha256'
                actual = base64.urlsafe_b64encode(hashlib.file_digest(path.open('rb'), 'sha256').digest()).rstrip(b'=').decode()
                assert actual == expected, path
                assert path.stat().st_size == int(size), path
            wheel.write(path, relative)
    print(f'Verified and cached {target.name}', flush=True)
