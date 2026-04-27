"""
JSON Model Downloader for AICoverGen
=====================================
Adapted from JSON-RVC-Inference (https://github.com/ArkanDash/JSON-RVC-Inference)

This module provides a centralized, JSON-driven model downloader that handles:
  - Core models (hubert_base.pt, rmvpe.pt)
  - MDX-Net vocal separation models (.onnx)
  - Voice models (.pth + .index, distributed as zip files)

Usage (CLI):
    python src/download_models.py                   # Download all required core + MDX-Net models
    python src/download_models.py --all             # Download everything including voice models
    python src/download_models.py --voice NAME      # Download a specific voice model from the manifest

Usage (module):
    from download_models import ModelDownloader
    dl = ModelDownloader()
    dl.download_core()                              # Download only core models
    dl.download_mdxnet()                            # Download only MDX-Net models
    dl.download_voice_model("Klee")                 # Download a specific voice model
"""

import glob
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

# Default directories
rvc_models_dir = BASE_DIR / 'rvc_models'
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'


class ModelDownloader:
    """
    JSON-driven model downloader for AICoverGen.

    Reads model definitions from models_manifest.json and downloads them
    to the appropriate directories. Supports core models, MDX-Net models,
    and RVC voice models (zip packages containing .pth and .index files).
    """

    def __init__(self, manifest_path=None):
        """
        Initialize the downloader by loading the JSON manifest.

        Args:
            manifest_path: Path to the JSON manifest file. If None, uses
                          the default models_manifest.json in the project root.
        """
        self.manifest_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
        self.manifest = self._load_manifest()
        self.temp_dir = None

    def _load_manifest(self):
        """Load and parse the JSON manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Model manifest not found at {self.manifest_path}. "
                "Please ensure models_manifest.json exists in the project root."
            )

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate manifest structure
        required_sections = ['core_models', 'mdxnet_models', 'voice_models']
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Manifest missing required section: '{section}'")

        return data

    def _download_file(self, url, dest_path, description="", chunk_size=8192):
        """
        Download a single file from URL to destination path with progress reporting.

        Args:
            url: The download URL
            dest_path: Full path to save the file (including filename)
            description: Human-readable description for progress output
            chunk_size: Download chunk size in bytes

        Returns:
            True if download succeeded, False otherwise
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists
        if dest_path.exists():
            file_size = dest_path.stat().st_size
            print(f"  [SKIP] {dest_path.name} already exists ({file_size / 1024 / 1024:.1f} MB)")
            return True

        desc_str = f"  [{description}] " if description else "  "
        print(f"{desc_str}Downloading {url.split('/')[-1]}...")

        try:
            # Handle Pixeldrain URLs - need to use the API endpoint
            download_url = url
            if 'pixeldrain.com' in url:
                file_id = url.rstrip('/').split('/')[-1]
                download_url = f'https://pixeldrain.com/api/file/{file_id}'

            with requests.get(download_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0

                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            print(f"\r{desc_str}Downloading {url.split('/')[-1]}... "
                                  f"{downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB "
                                  f"({percent:.1f}%)", end='', flush=True)

                print()  # New line after progress

            # Verify download
            if dest_path.exists() and dest_path.stat().st_size > 0:
                size_mb = dest_path.stat().st_size / 1024 / 1024
                print(f"  [DONE] {dest_path.name} ({size_mb:.1f} MB)")
                return True
            else:
                print(f"  [ERROR] Downloaded file is empty: {dest_path.name}")
                dest_path.unlink(missing_ok=True)
                return False

        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Failed to download {url}: {e}")
            dest_path.unlink(missing_ok=True)
            return False

    def download_core(self):
        """
        Download all required core models (hubert_base.pt, rmvpe.pt).

        Core models are saved directly to the rvc_models/ directory.
        """
        print("=" * 60)
        print("Downloading Core Models")
        print("=" * 60)

        core_models = self.manifest.get('core_models', {})
        if not core_models:
            print("  No core models defined in manifest.")
            return True

        success = True
        for filename, info in core_models.items():
            dest_dir = BASE_DIR / info['dest']
            dest_path = dest_dir / filename
            if not self._download_file(
                info['url'], dest_path,
                description=f"Core: {info.get('description', filename)}"
            ):
                success = False

        return success

    def download_mdxnet(self):
        """
        Download all required MDX-Net vocal separation models (.onnx).

        MDX-Net models are saved directly to the mdxnet_models/ directory.
        """
        print("=" * 60)
        print("Downloading MDX-Net Models")
        print("=" * 60)

        mdxnet_models = self.manifest.get('mdxnet_models', {})
        if not mdxnet_models:
            print("  No MDX-Net models defined in manifest.")
            return True

        success = True
        for filename, info in mdxnet_models.items():
            dest_dir = BASE_DIR / info['dest']
            dest_path = dest_dir / filename
            if not self._download_file(
                info['url'], dest_path,
                description=f"MDX-Net: {info.get('description', filename)}"
            ):
                success = False

        return success

    def get_voice_model_list(self):
        """
        Get the list of available voice model names from the manifest.

        Returns:
            List of voice model name strings.
        """
        voice_models = self.manifest.get('voice_models', {})
        return voice_models.get('list', [])

    def download_voice_model(self, model_name, progress_callback=None):
        """
        Download a voice model by name from the manifest.

        Adapted from JSON-RVC-Inference's download_model() function.
        Downloads a zip file containing .pth and .index files, extracts
        them, and places them in the appropriate rvc_models/<name>/ directory.

        Args:
            model_name: The display name of the voice model (must match manifest)
            progress_callback: Optional callback function for progress updates.
                             Signature: callback(message: str)

        Returns:
            str: Success/error message
        """
        voice_models = self.manifest.get('voice_models', {})
        model_data = voice_models.get('model_data', [])
        model_list = voice_models.get('list', [])

        if model_name not in model_list:
            return f"[ERROR] Voice model '{model_name}' not found in manifest."

        # Check if model directory already exists
        extraction_folder = rvc_models_dir / model_name
        if extraction_folder.exists():
            return f"[SKIP] Voice model directory '{model_name}' already exists."

        # Find the model entry in model_data
        matched_item = None
        for item in model_data:
            # item[1] is the model name
            if len(item) >= 3 and item[1] == model_name:
                matched_item = item
                break

        if matched_item is None:
            return f"[ERROR] Voice model '{model_name}' found in list but has no download data."

        if progress_callback:
            progress_callback(f"[~] Downloading voice model '{model_name}'...")

        # item[2] is the zip URL
        zip_url = matched_item[2]

        try:
            # Use a temp directory for download and extraction
            self.temp_dir = tempfile.mkdtemp(prefix='aicovergen_dl_')

            # Handle Pixeldrain URLs
            download_url = zip_url
            if 'pixeldrain.com' in zip_url:
                file_id = zip_url.rstrip('/').split('/')[-1]
                download_url = f'https://pixeldrain.com/api/file/{file_id}'

            # Download zip
            zip_path = os.path.join(self.temp_dir, f'{model_name}.zip')
            if progress_callback:
                progress_callback(f"[~] Fetching model archive...")

            with requests.get(download_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0

                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            msg = f"[~] Downloading... {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)"
                            if progress_callback:
                                progress_callback(msg)

            if progress_callback:
                progress_callback("[~] Extracting zip archive...")

            # Extract zip
            os.makedirs(extraction_folder, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)

            # Remove the zip file after extraction
            os.remove(zip_path)

            # Find and move .pth and .index files
            pth_found = False
            index_found = False

            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    src_path = os.path.join(root, file)

                    if file.endswith('.pth') and os.stat(src_path).st_size > 1024 * 1024 * 40:
                        dest = os.path.join(extraction_folder, os.path.basename(src_path))
                        shutil.move(src_path, dest)
                        pth_found = True

                    elif file.endswith('.index') and os.stat(src_path).st_size > 1024 * 100:
                        dest = os.path.join(extraction_folder, os.path.basename(src_path))
                        shutil.move(src_path, dest)
                        index_found = True

            # Clean up nested directories
            for filepath in os.listdir(extraction_folder):
                if os.path.isdir(os.path.join(extraction_folder, filepath)):
                    shutil.rmtree(os.path.join(extraction_folder, filepath))

            if not pth_found:
                # Clean up and report error
                shutil.rmtree(extraction_folder, ignore_errors=True)
                return f"[ERROR] No .pth model file found in the downloaded zip for '{model_name}'."

            # Clean up temp directory
            self._cleanup_temp()

            status_parts = [f"[+] {model_name} model successfully downloaded!"]
            if index_found:
                status_parts.append(" (with index file)")
            return "".join(status_parts)

        except requests.exceptions.RequestException as e:
            self._cleanup_temp()
            if extraction_folder.exists():
                shutil.rmtree(extraction_folder, ignore_errors=True)
            return f"[ERROR] Download failed for '{model_name}': {e}"

        except Exception as e:
            self._cleanup_temp()
            if extraction_folder.exists():
                shutil.rmtree(extraction_folder, ignore_errors=True)
            return f"[ERROR] Failed to process '{model_name}': {e}"

    def download_all_voice_models(self, progress_callback=None):
        """
        Download all voice models from the manifest.

        Args:
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple of (success_count, fail_count, messages)
        """
        model_list = self.get_voice_model_list()
        success_count = 0
        fail_count = 0
        messages = []

        total = len(model_list)
        for i, name in enumerate(model_list):
            if progress_callback:
                progress_callback(f"\n[{i + 1}/{total}] Processing '{name}'...")

            result = self.download_voice_model(name, progress_callback)
            messages.append(f"  {name}: {result}")

            if result.startswith("[+]"):
                success_count += 1
            elif result.startswith("[SKIP]"):
                success_count += 1  # Already existing is OK
            else:
                fail_count += 1

        return success_count, fail_count, messages

    def check_existing(self):
        """
        Check which required models are already downloaded.

        Returns:
            dict with 'core', 'mdxnet', and 'voice' keys, each containing
            a dict of filename -> bool (True if exists)
        """
        result = {
            'core': {},
            'mdxnet': {},
            'voice': {}
        }

        # Check core models
        for filename, info in self.manifest.get('core_models', {}).items():
            dest_dir = BASE_DIR / info['dest']
            dest_path = dest_dir / filename
            result['core'][filename] = dest_path.exists()

        # Check MDX-Net models
        for filename, info in self.manifest.get('mdxnet_models', {}).items():
            dest_dir = BASE_DIR / info['dest']
            dest_path = dest_dir / filename
            result['mdxnet'][filename] = dest_path.exists()

        # Check voice models
        model_list = self.get_voice_model_list()
        for name in model_list:
            model_dir = rvc_models_dir / name
            result['voice'][name] = model_dir.exists()

        return result

    def download_required(self):
        """
        Download all required (non-voice) models that are missing.

        Returns:
            True if all required models are now available.
        """
        print("Checking existing models...")
        existing = self.check_existing()

        needs_core = not all(existing['core'].values())
        needs_mdxnet = not all(existing['mdxnet'].values())

        success = True

        if needs_core:
            print("\nMissing core models detected.")
            if not self.download_core():
                success = False

        if needs_mdxnet:
            print("\nMissing MDX-Net models detected.")
            if not self.download_mdxnet():
                success = False

        if not needs_core and not needs_mdxnet:
            print("All required models are already present!")

        return success

    def _cleanup_temp(self):
        """Remove temporary download directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


# ============================================================
# CLI Entry Point
# ============================================================

def print_usage():
    """Print CLI usage information."""
    usage = """
AICoverGen JSON Model Downloader
=================================

Usage:
    python src/download_models.py                   Download all required core + MDX-Net models
    python src/download_models.py --all             Download everything including voice models
    python src/download_models.py --core            Download only core models (hubert, rmvpe)
    python src/download_models.py --mdxnet          Download only MDX-Net models
    python src/download_models.py --voice NAME      Download a specific voice model
    python src/download_models.py --list            List all available voice models
    python src/download_models.py --check           Check which models are already downloaded

Options:
    --all       Download all models (core + MDX-Net + all voice models)
    --core      Download only core models (hubert_base.pt, rmvpe.pt)
    --mdxnet    Download only MDX-Net vocal separation models
    --voice     Download a specific voice model by name (e.g., --voice "Klee")
    --list      List all voice models available in the manifest
    --check     Check which required models are already downloaded
    --help      Show this help message

Manifest: models_manifest.json (in project root)
"""
    print(usage)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h'):
        print_usage()
        sys.exit(0)

    downloader = ModelDownloader()
    command = sys.argv[1]

    if command == '--check':
        existing = downloader.check_existing()
        print("\nCore Models:")
        for name, exists in existing['core'].items():
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {name}")

        print("\nMDX-Net Models:")
        for name, exists in existing['mdxnet'].items():
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {name}")

        print(f"\nVoice Models: {sum(existing['voice'].values())}/{len(existing['voice'])} downloaded")

    elif command == '--list':
        models = downloader.get_voice_model_list()
        print(f"\nAvailable Voice Models ({len(models)} total):")
        for i, name in enumerate(models, 1):
            model_dir = rvc_models_dir / name
            status = "[OWNED]" if model_dir.exists() else ""
            print(f"  {i:3d}. {name} {status}")

    elif command == '--core':
        downloader.download_core()

    elif command == '--mdxnet':
        downloader.download_mdxnet()

    elif command == '--voice':
        if len(sys.argv) < 3:
            print("Error: Please specify a model name. Example: --voice \"Klee\"")
            print("Use --list to see available models.")
            sys.exit(1)
        model_name = sys.argv[2]
        result = downloader.download_voice_model(model_name)
        print(result)

    elif command == '--all':
        # Download required models first
        downloader.download_required()

        # Then download all voice models
        print("\n" + "=" * 60)
        print("Downloading All Voice Models")
        print("=" * 60)
        success, fail, messages = downloader.download_all_voice_models()
        print(f"\n{'=' * 60}")
        print(f"Voice model download complete: {success} succeeded, {fail} failed")
        print(f"{'=' * 60}")

    else:
        # Default: download all required models
        downloader.download_required()
