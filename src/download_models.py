"""
JSON Model Downloader for AICoverGen
=====================================
Adapted from JSON-RVC-Inference (https://github.com/ArkanDash/JSON-RVC-Inference)

Two JSON files:
  - models_manifest.json  (root)  : hubert_base.pt, rmvpe.pt, MDX-Net .onnx models
  - rvc_models/list.json          : voice model list + zip download URLs

CLI:
    python src/download_models.py                   # download missing required models
    python src/download_models.py --voice NAME      # download a voice model
    python src/download_models.py --check           # check model status
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

rvc_models_dir = BASE_DIR / 'rvc_models'


def _load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _download(url, dest, label=""):
    """Download a file to dest. Skip if already exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [SKIP] {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    tag = f"[{label}] " if label else ""
    dl_url = url
    if 'pixeldrain.com' in url:
        dl_url = f"https://pixeldrain.com/api/file/{url.rstrip('/').split('/')[-1]}"

    print(f"  {tag}Downloading {dest.name}...", end='', flush=True)
    try:
        with requests.get(dl_url, stream=True, timeout=60) as r:
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


def download_required():
    """Download all required models from models_manifest.json."""
    manifest = _load_json(MANIFEST_PATH)
    print("Downloading required models...")
    ok = True
    for rel_path, url in manifest.items():
        dest = BASE_DIR / rel_path
        label = Path(rel_path).parent.name
        if not _download(url, dest, label=label):
            ok = False
    if ok:
        print("All required models ready.")
    return ok


def check_status():
    """Print status of required and voice models."""
    manifest = _load_json(MANIFEST_PATH)
    voice_data = _load_json(VOICE_LIST_PATH)

    print("\nRequired models:")
    for rel_path in manifest:
        name = Path(rel_path).name
        exists = (BASE_DIR / rel_path).exists()
        print(f"  {'[OK]' if exists else '[MISSING]'} {name}")

    voice_list = voice_data.get('list', [])
    owned = sum(1 for n in voice_list if (rvc_models_dir / n).exists())
    print(f"\nVoice models: {owned}/{len(voice_list)} downloaded")


def get_voice_list():
    """Return the voice model name list from list.json."""
    return _load_json(VOICE_LIST_PATH).get('list', [])


def download_voice_model(model_name, progress_callback=None):
    """Download a voice model by name. Adapted from JSON-RVC-Inference."""
    voice_data = _load_json(VOICE_LIST_PATH)
    model_list = voice_data.get('list', [])
    model_data = voice_data.get('model_data', [])

    if model_name not in model_list:
        return f"[ERROR] '{model_name}' not found in rvc_models/list.json"

    dest = rvc_models_dir / model_name
    if dest.exists():
        return f"[SKIP] '{model_name}' already exists"

    # find entry: item[1] == model_name, item[2] == zip url
    zip_url = None
    for item in model_data:
        if len(item) >= 3 and item[1] == model_name:
            zip_url = item[2]
            break
    if not zip_url:
        return f"[ERROR] '{model_name}' has no download URL in list.json"

    if progress_callback:
        progress_callback(f"[~] Downloading '{model_name}'...")

    try:
        tmp = tempfile.mkdtemp(prefix='aicovergen_dl_')
        dl_url = zip_url
        if 'pixeldrain.com' in zip_url:
            dl_url = f"https://pixeldrain.com/api/file/{zip_url.rstrip('/').split('/')[-1]}"

        zip_path = os.path.join(tmp, f'{model_name}.zip')
        if progress_callback:
            progress_callback(f"[~] Fetching zip...")

        with requests.get(dl_url, stream=True, timeout=60) as r:
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
        names = get_voice_list()
        for i, n in enumerate(names, 1):
            tag = " [OWNED]" if (rvc_models_dir / n).exists() else ""
            print(f"  {i:3d}. {n}{tag}")

    elif cmd == '--voice':
        if len(sys.argv) < 3:
            print("Error: specify model name. Example: --voice \"Klee\"")
            sys.exit(1)
        print(download_voice_model(sys.argv[2]))

    else:
        download_required()
