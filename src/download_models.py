"""
JSON Model Downloader for AICoverGen
=====================================

Two JSON files:
  - models_manifest.json  (root)  : hubert_base.pt, rmvpe.pt, MDX-Net .onnx
  - rvc_models/list.json          : voice models (name, url, image, description, credit)

Usage:
    python src/download_models.py                   download required models
    python src/download_models.py --voice NAME      download a voice model
    python src/download_models.py --check           check model status
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / 'models_manifest.json'
VOICE_LIST_PATH = BASE_DIR / 'rvc_models' / 'list.json'
DEFAULT_IMAGE = BASE_DIR / 'images' / 'default_model.png'
RVC_MODELS_DIR = BASE_DIR / 'rvc_models'


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _pixeldrain_url(url):
    """Convert pixeldrain page URL to API download URL."""
    if 'pixeldrain.com' in url:
        return f"https://pixeldrain.com/api/file/{url.rstrip('/').split('/')[-1]}"
    return url


def _download_file(url, dest, label=""):
    """Download a file to dest. Skip if already exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [SKIP] {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    tag = f"[{label}] " if label else ""
    print(f"  {tag}Downloading {dest.name}...", end='', flush=True)
    try:
        with requests.get(_pixeldrain_url(url), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        size = dest.stat().st_size
        if size == 0:
            dest.unlink(missing_ok=True)
            print(" FAILED (empty)")
            return False
        print(f" OK ({size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        dest.unlink(missing_ok=True)
        print(f" FAILED ({e})")
        return False


# ── Required models (hubert / rmvpe / mdxnet) ─────────────────────────

def download_required():
    """Download all required models from models_manifest.json."""
    manifest = _load_json(MANIFEST_PATH)
    print("Downloading required models...")
    ok = all(_download_file(url, BASE_DIR / rel, Path(rel).parent.name)
             for rel, url in manifest.items())
    print("All required models ready." if ok else "Some downloads failed.")
    return ok


# ── Voice models ──────────────────────────────────────────────────────

def get_voice_list():
    """Return list of voice model dicts from list.json."""
    return _load_json(VOICE_LIST_PATH)


def get_voice_names():
    """Return list of voice model names."""
    return [m['name'] for m in get_voice_list()]


def get_voice_model(model_name):
    """Get a voice model dict by name, or None."""
    for m in get_voice_list():
        if m['name'] == model_name:
            return m
    return None


def get_model_image(model_name):
    """Return image URL/path for a model. Falls back to default placeholder."""
    model = get_voice_model(model_name)
    if not model:
        return str(DEFAULT_IMAGE)
    image = model.get('image', '')
    if image and image.startswith('http'):
        return image
    return str(DEFAULT_IMAGE)


def download_voice_model(model_name, progress_callback=None):
    """Download and extract a voice model by name."""
    model = get_voice_model(model_name)
    if not model:
        return f"[ERROR] '{model_name}' not found in list.json"

    url = model.get('url', '')
    if not url:
        return f"[ERROR] '{model_name}' has no download URL"

    dest = RVC_MODELS_DIR / model_name
    if dest.exists():
        return f"[SKIP] '{model_name}' already exists"

    if progress_callback:
        progress_callback(f"[~] Downloading '{model_name}'...")

    try:
        tmp = tempfile.mkdtemp(prefix='aicovergen_dl_')
        zip_path = os.path.join(tmp, f'{model_name}.zip')

        if progress_callback:
            progress_callback("[~] Fetching zip...")

        with requests.get(_pixeldrain_url(url), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if progress_callback:
            progress_callback("[~] Extracting...")

        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmp)
        os.remove(zip_path)

        pth_found = index_found = False
        for root, _, files in os.walk(tmp):
            for f in files:
                src = os.path.join(root, f)
                if f.endswith('.pth') and os.stat(src).st_size > 40 * 1024 * 1024:
                    shutil.move(src, os.path.join(dest, f))
                    pth_found = True
                elif f.endswith('.index') and os.stat(src).st_size > 100 * 1024:
                    shutil.move(src, os.path.join(dest, f))
                    index_found = True

        # cleanup nested dirs inside dest
        for p in os.listdir(dest):
            if os.path.isdir(os.path.join(dest, p)):
                shutil.rmtree(os.path.join(dest, p))

        shutil.rmtree(tmp, ignore_errors=True)

        if not pth_found:
            shutil.rmtree(dest, ignore_errors=True)
            return f"[ERROR] No .pth found in zip for '{model_name}'"

        return f"[+] '{model_name}' downloaded!" + (" (with index)" if index_found else "")

    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return f"[ERROR] '{model_name}': {e}"


# ── Status check ──────────────────────────────────────────────────────

def check_status():
    """Print status of all models."""
    manifest = _load_json(MANIFEST_PATH)
    voice_list = get_voice_list()

    print("\nRequired models:")
    for rel_path in manifest:
        name = Path(rel_path).name
        exists = (BASE_DIR / rel_path).exists()
        print(f"  {'[OK]' if exists else '[MISSING]'} {name}")

    owned = sum(1 for m in voice_list if (RVC_MODELS_DIR / m['name']).exists())
    print(f"\nVoice models: {owned}/{len(voice_list)} downloaded")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''

    if cmd in ('-h', '--help'):
        print("Usage:\n"
              "  python src/download_models.py              download required models\n"
              "  python src/download_models.py --voice NAME download voice model\n"
              "  python src/download_models.py --check      check model status\n"
              "  python src/download_models.py --list       list voice models")
    elif cmd == '--check':
        check_status()
    elif cmd == '--list':
        for i, m in enumerate(get_voice_list(), 1):
            tag = " [OWNED]" if (RVC_MODELS_DIR / m['name']).exists() else ""
            print(f"  {i:3d}. {m['name']}{tag}  - {m.get('description', '')}")
    elif cmd == '--voice':
        if len(sys.argv) < 3:
            print("Error: specify model name. Example: --voice \"Klee\"")
            sys.exit(1)
        print(download_voice_model(sys.argv[2]))
    else:
        download_required()
