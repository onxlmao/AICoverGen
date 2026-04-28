---
Task ID: 1
Agent: Super Z (main)
Task: Clone AICoverGen and JSON-RVC-Inference repositories, then adapt the JSON model downloader

Work Log:
- Cloned https://github.com/onxlmao/AICoverGen.git to /home/z/my-project/AICoverGen
- Cloned https://github.com/ArkanDash/JSON-RVC-Inference.git to /home/z/my-project/JSON-RVC-Inference
- Explored both repositories to understand their architecture and model download mechanisms
- JSON-RVC-Inference uses a JSON manifest with `list` (model names) and `model_data` (download entries with zip URL + cover URL), `download_model()` function uses `wget` + `zipfile`
- AICoverGen had a basic `download_models.py` with only 2 hardcoded models (hubert_base.pt, rmvpe.pt), no MDX-Net download mechanism, and duplicate download logic in webui.py
- Created `models_manifest.json` - comprehensive JSON manifest covering core models (hubert, rmvpe), MDX-Net models (3 .onnx files), and 55 voice models (migrated from public_models.json into JSON-RVC-Inference compatible format)
- Rewrote `src/download_models.py` as a full `ModelDownloader` class with: download_core(), download_mdxnet(), download_voice_model(), check_existing(), download_required(), download_all_voice_models() methods; CLI interface with --core, --mdxnet, --voice, --all, --list, --check flags; progress reporting, error handling, skip-if-exists, Pixeldrain URL support
- Updated `src/webui.py` to import ModelDownloader and add a new "From JSON Index" sub-tab under the existing "Download model" tab, with dropdown selection, one-click voice model download, required models downloader, and model status checker

Stage Summary:
- 3 files created/modified in AICoverGen: models_manifest.json (new), src/download_models.py (rewritten), src/webui.py (updated)
- All tests pass: manifest loads correctly (55 voice models, 2 core, 3 MDX-Net), CLI --check/--list/--help work
- Voice model format is backward-compatible with JSON-RVC-Inference's `model_data` array format
- Existing WebUI functionality (Generate, Upload model, From HuggingFace/Pixeldrain URL, From Public Index) fully preserved
