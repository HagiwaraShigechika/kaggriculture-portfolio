"""Paths and separate output directories shared by evaluation commands."""
from datetime import datetime
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def input_path(value):
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def output_directory(label, requested=None):
    if requested is not None:
        directory = input_path(requested)
        directory.mkdir(parents=True, exist_ok=False)
        return directory
    parent = ROOT / 'results' / 'runs'
    parent.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime('%Y%m%d-%H%M%S-') + label + '-'
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
