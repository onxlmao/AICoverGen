# AICoverGen

An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio file. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

Showcase: https://www.youtube.com/watch?v=2qZuE4WM7CM

Setup Guide: https://www.youtube.com/watch?v=pdlhk4vVHQk

![](images/webui_generate.png?raw=true)

WebUI is under constant development and testing, but you can try it out right now on both local and colab!

## Table of Contents

- [Changelog](#changelog)
- [Update AICoverGen to latest version](#update-aicovergen-to-latest-version)
- [Colab notebook](#colab-notebook)
- [Setup](#setup)
    - [Install Git and Python](#install-git-and-python)
    - [Install ffmpeg and sox](#install-ffmpeg)
    - [Clone AICoverGen repository](#clone-aicovergen-repository)
    - [Download required models](#download-required-models)
- [Usage with WebUI](#usage-with-webui)
    - [Download RVC models via JSON Gallery](#download-rvc-models-via-json-gallery)
    - [Download RVC models via URL](#download-rvc-models-via-url)
    - [Download RVC models from Public Index](#download-rvc-models-from-public-index)
    - [Upload RVC models via WebUI](#upload-rvc-models-via-webui)
    - [Running the pipeline via WebUI](#running-the-pipeline-via-webui)
- [Usage with CLI](#usage-with-cli)
    - [Manual Download of RVC models](#manual-download-of-rvc-models)
    - [Running the pipeline via CLI](#running-the-pipeline-via-cli)
- [Model Management System](#model-management-system)
    - [models_manifest.json](#models_manifestjson)
    - [rvc_models/list.json](#rvc_modelslistjson)
    - [Adding your own models to the gallery](#adding-your-own-models-to-the-gallery)
- [Directory Structure](#directory-structure)
- [Terms of Use](#terms-of-use)


## Changelog

### Recent Updates

- **JSON-based model management system** - Download voice models from a curated gallery with images, descriptions, and credits via `rvc_models/list.json`
- **Model Gallery with click-to-select** - Browse all 55 voice models in a visual gallery grid; click any model card to instantly select it in the dropdown
- **models_manifest.json** - Centralized manifest for required models (hubert, rmvpe, MDX-Net) with one-click download
- **Skip-if-exists for audio separation** - Previously separated audio files are reused automatically, saving processing time
- **Model status checking** - Verify which required and voice models are downloaded with a single click
- **Image caching** - Model images are downloaded locally on first use to avoid hotlink protection issues
- **Gradio 6.x compatibility** - Full support for latest Gradio version with proper `gr.update()` API

### Original Features

- WebUI for easier conversions and downloading of voice models
- Support for cover generations from a local audio file
- Option to keep intermediate files generated. e.g. Isolated vocals/instrumentals
- Download suggested public voice models from table with search/tag filters
- Support for Pixeldrain download links for voice models
- Implement new rmvpe pitch extraction technique for faster and higher quality vocal conversions
- Volume control for AI main vocals, backup vocals and instrumentals
- Index Rate for Voice conversion
- Reverb Control for AI main vocals
- Local network sharing option for webui
- Extra RVC options - filter_radius, rms_mix_rate, protect
- Local file upload via file browser option
- Upload of locally trained RVC v2 models via WebUI
- Pitch detection method control, e.g. rmvpe/mangio-crepe
- Pitch change for vocals and instrumentals together. Same effect as changing key of song in Karaoke.
- Audio output format option: wav or mp3.


## Update AICoverGen to latest version

Install and pull any new requirements and changes by opening a command line window in the `AICoverGen` directory and running the following commands.

```
pip install -r requirements.txt
git pull
```

For colab users, simply click `Runtime` in the top navigation bar of the colab notebook and `Disconnect and delete runtime` in the dropdown menu.
Then follow the instructions in the notebook to run the webui.

## Colab notebook

For those without a powerful enough NVIDIA GPU, you may try AICoverGen out using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onxlmao/AICoverGen/blob/main/AICoverGen_colab.ipynb)

For those who face issues with Google Colab notebook disconnecting after a few minutes, here's an alternative that doesn't use the WebUI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ardha27/AICoverGen-NoUI-Colab/blob/main/CoverGen_No_UI.ipynb)

For those who want to run this locally, follow the setup guide below.

## Setup

### Install Git and Python

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer. Also follow this [guide](https://realpython.com/installing-python/) to install Python **VERSION 3.9** if you haven't already. Using other versions of Python may result in dependency conflicts.

### Install ffmpeg

Follow the instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) to install ffmpeg on your computer.

### Install sox

Follow the instructions [here](https://www.tutorialexample.com/a-step-guide-to-install-sox-sound-exchange-on-windows-10-python-tutorial/) to install sox and add it to your Windows path environment.

### Clone AICoverGen repository

Open a command line window and run these commands to clone this entire repository and install the additional dependencies required.

```
git clone https://github.com/onxlmao/AICoverGen
cd AICoverGen
pip install -r requirements.txt
```

### Download required models

Run the following command to download the required MDXNET vocal separation models and hubert base model.

```
python src/download_models.py
```

This reads from `models_manifest.json` and downloads:
- `hubert_base.pt` - Base model for voice conversion
- `rmvpe.pt` - Pitch extraction model
- MDX-Net models for vocal separation (`UVR-MDX-NET-Voc_FT.onnx`, `UVR_MDXNET_KARA_2.onnx`, `Reverb_HQ_By_FoxJoy.onnx`)

Files that already exist will be skipped automatically. You can check the status at any time:

```
python src/download_models.py --check
```


## Usage with WebUI

To run the AICoverGen WebUI, run the following command.

```
python src/webui.py
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--share`                                  | Create a public URL. This is useful for running the web UI on Google Colab. |
| `--listen`                                 | Make the web UI reachable from your local network. |
| `--listen-host LISTEN_HOST`                | The hostname that the server will use. |
| `--listen-port LISTEN_PORT`                | The listening port that the server will use. |

Once the following output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the WebUI.

### Download RVC models via JSON Gallery

![](images/webui_dl_model.png?raw=true)

Navigate to the `Download model` tab and open the **From JSON Index** sub-tab. Here you will find:

1. **Model Gallery** - A visual grid of all available voice models with character images. Each card shows the model name and a status indicator:
   - ✅ = Already downloaded and ready to use
   - ⬇️ = Not yet downloaded
2. **Click to Select** - Click any model card in the gallery to instantly select it in the dropdown below. The preview panel on the right will show the model's image, description, credit, and download status.
3. **Dropdown** - Alternatively, select a model from the dropdown list to browse details before downloading.
4. **Download Model** - Click the button to download the selected voice model. Once downloaded, it becomes available in the `Generate` tab after clicking `Refresh Models`.
5. **Download Required Models** - One-click download for all required base models (hubert, rmvpe, MDX-Net) from `models_manifest.json`.
6. **Check Model Status** - View which required models and voice models are present on your system.

The gallery currently includes **55 voice models** from anime, games, and real artists, each with character artwork sourced from AniList, Enka Network, and Zerochan. Image files are cached locally in `images/models/` after first download.

### Download RVC models via URL

![](images/webui_dl_model.png?raw=true)

Navigate to the `Download model` tab and open the **From HuggingFace/Pixeldrain URL** sub-tab. Paste the download link to the RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download. You may refer to the examples for how the download link should look like.
The downloaded zip file should contain the .pth model file and an optional .index file.

### Download RVC models from Public Index

Navigate to the `Download model` tab and open the **From Public Index** sub-tab. Click `Initialize public models table` to load the available models. Use tags and search to filter the list. Click a row to autofill the download link and model name, then click `Download`.

### Upload RVC models via WebUI

![](images/webui_upload_model.png?raw=true)

For people who have trained RVC v2 models locally and would like to use them for AI Cover generations.
Navigate to the `Upload model` tab, and follow the instructions.
Once the output message says `[NAME] Model successfully uploaded!`, you should be able to use it in the `Generate` tab after clicking the refresh models button!


### Running the pipeline via WebUI

![](images/webui_generate.png?raw=true)

- From the Voice Models dropdown menu, select the voice model to use. Click `Refresh Models` if you added the files manually to the [rvc_models](rvc_models) directory to refresh the list.
- In the song input field, copy and paste the link to any song on YouTube or the full path to a local audio file.
- Pitch should be set to either -12, 0, or 12 depending on the original vocals and the RVC AI model. This ensures the voice is not *out of tune*.
- Other advanced options for Voice conversion and audio mixing can be viewed by clicking the accordion arrow to expand.
- **Note:** If audio has already been separated for the same input song, it will be reused automatically to save processing time.

Once all Main Options are filled in, click `Generate` and the AI generated cover should appear in a less than a few minutes depending on your GPU.

## Usage with CLI

### Manual Download of RVC models

Unzip (if needed) and transfer the `.pth` and `.index` files to a new folder in the [rvc_models](rvc_models) directory. Each folder should only contain one `.pth` and one `.index` file.

You can also download models directly using the CLI:

```bash
# Download a specific voice model by name
python src/download_models.py --voice "Klee"

# List all available voice models
python src/download_models.py --list

# Check which models are downloaded
python src/download_models.py --check
```

### Running the pipeline via CLI

To run the AI cover generation pipeline using the command line, run the following command.

```
python src/main.py [-h] -i SONG_INPUT -dir RVC_DIRNAME -p PITCH_CHANGE [-k | --keep-files | --no-keep-files] [-ir INDEX_RATE] [-fr FILTER_RADIUS] [-rms RMS_MIX_RATE] [-palgo PITCH_DETECTION_ALGO] [-hop CREPE_HOP_LENGTH] [-pro PROTECT] [-mv MAIN_VOL] [-bv BACKUP_VOL] [-iv INST_VOL] [-pall PITCH_CHANGE_ALL] [-rsize REVERB_SIZE] [-rwet REVERB_WETNESS] [-rdry REVERB_DRYNESS] [-rdamp REVERB_DAMPING] [-oformat OUTPUT_FORMAT]
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `-i SONG_INPUT`                            | Link to a song on YouTube or path to a local audio file. Should be enclosed in double quotes for Windows and single quotes for Unix-like systems. |
| `-dir MODEL_DIR_NAME`                      | Name of folder in [rvc_models](rvc_models) directory containing your `.pth` and `.index` files for a specific voice. |
| `-p PITCH_CHANGE`                          | Change pitch of AI vocals in octaves. Set to 0 for no change. Generally, use 1 for male to female conversions and -1 for vice-versa. |
| `-k`                                       | Optional. Can be added to keep all intermediate audio files generated. e.g. Isolated AI vocals/instrumentals. Leave out to save space. |
| `-ir INDEX_RATE`                           | Optional. Default 0.5. Control how much of the AI's accent to leave in the vocals. 0 <= INDEX_RATE <= 1. |
| `-fr FILTER_RADIUS`                        | Optional. Default 3. If >=3: apply median filtering median filtering to the harvested pitch results. 0 <= FILTER_RADIUS <= 7. |
| `-rms RMS_MIX_RATE`                        | Optional. Default 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1). 0 <= RMS_MIX_RATE <= 1. |
| `-palgo PITCH_DETECTION_ALGO`              | Optional. Default rmvpe. Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals). |
| `-hop CREPE_HOP_LENGTH`                    | Optional. Default 128. Controls how often it checks for pitch changes in milliseconds when using mangio-crepe algo specifically. Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy. |
| `-pro PROTECT`                             | Optional. Default 0.33. Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable. 0 <= PROTECT <= 0.5. |
| `-mv MAIN_VOCALS_VOLUME_CHANGE`            | Optional. Default 0. Control volume of main AI vocals. Use -3 to decrease the volume by 3 decibels, or 3 to increase the volume by 3 decibels. |
| `-bv BACKUP_VOCALS_VOLUME_CHANGE`          | Optional. Default 0. Control volume of backup AI vocals. |
| `-iv INSTRUMENTAL_VOLUME_CHANGE`           | Optional. Default 0. Control volume of the background music/instrumentals. |
| `-pall PITCH_CHANGE_ALL`                   | Optional. Default 0. Change pitch/key of background music, backup vocals and AI vocals in semitones. Reduces sound quality slightly. |
| `-rsize REVERB_SIZE`                       | Optional. Default 0.15. The larger the room, the longer the reverb time. 0 <= REVERB_SIZE <= 1. |
| `-rwet REVERB_WETNESS`                     | Optional. Default 0.2. Level of AI vocals with reverb. 0 <= REVERB_WETNESS <= 1. |
| `-rdry REVERB_DRYNESS`                     | Optional. Default 0.8. Level of AI vocals without reverb. 0 <= REVERB_DRYNESS <= 1. |
| `-rdamp REVERB_DAMPING`                    | Optional. Default 0.7. Absorption of high frequencies in the reverb. 0 <= REVERB_DAMPING <= 1. |
| `-oformat OUTPUT_FORMAT`                   | Optional. Default mp3. wav for best quality and large file size, mp3 for decent quality and small file size. |


## Model Management System

AICoverGen uses a JSON-based model management system for easy downloading and organization of voice models and required base models.

### models_manifest.json

Located in the project root, this file defines the required base models needed for the pipeline to function:

```json
{
    "hubert_base.pt": "https://huggingface.co/.../hubert_base.pt",
    "rmvpe.pt": "https://huggingface.co/.../rmvpe.pt",
    "mdxnet_models/UVR-MDX-NET-Voc_FT.onnx": "https://huggingface.co/.../UVR-MDX-NET-Voc_FT.onnx",
    "mdxnet_models/UVR_MDXNET_KARA_2.onnx": "https://huggingface.co/.../UVR_MDXNET_KARA_2.onnx",
    "mdxnet_models/Reverb_HQ_By_FoxJoy.onnx": "https://huggingface.co/.../Reverb_HQ_By_FoxJoy.onnx"
}
```

Each key is the relative path from the project root where the file will be saved. The value is the download URL. These models are downloaded to the project root directory (not inside `rvc_models/`).

### rvc_models/list.json

This file contains the curated list of downloadable voice models displayed in the gallery. Each entry is a flat JSON object:

```json
[
    {
        "name": "Klee",
        "url": "https://huggingface.co/qweshkka/Klee/resolve/main/Klee.zip",
        "image": "https://enka.network/ui/UI_AvatarIcon_Klee.png",
        "description": "Klee from Genshin Impact",
        "credit": "qweshsmashjuicefruity"
    }
]
```

| Field          | Required | Description |
|----------------|----------|-------------|
| `name`         | Yes      | Unique identifier for the model. Also used as the folder name in `rvc_models/`. |
| `url`          | Yes      | Direct download URL to a zip file containing `.pth` (and optionally `.index`) files. Pixeldrain URLs are auto-converted. |
| `image`        | No       | URL to a character/model image for the gallery. Falls back to `images/default_model.png` if empty or broken. |
| `description`  | No       | Brief description of the model/character shown in the preview panel. |
| `credit`       | No       | Credits the original model trainer/creator. |

### Adding your own models to the gallery

To add a new model to the gallery, simply append a new entry to `rvc_models/list.json`:

```json
{
    "name": "YourModelName",
    "url": "https://huggingface.co/your-repo/resolve/main/YourModel.zip",
    "image": "https://example.com/your-model-image.jpg",
    "description": "Description of your model",
    "credit": "YourName"
}
```

The model will automatically appear in the gallery on the next WebUI launch. Image URLs from AniList, Enka Network, and Zerochan are supported. Images are cached locally in `images/models/` after the first download.

## Directory Structure

```
AICoverGen/
├── models_manifest.json          # Required models (hubert, rmvpe, MDX-Net)
├── hubert_base.pt                # Downloaded by models_manifest.json
├── rmvpe.pt                      # Downloaded by models_manifest.json
├── mdxnet_models/                # Downloaded by models_manifest.json
│   ├── UVR-MDX-NET-Voc_FT.onnx
│   ├── UVR_MDXNET_KARA_2.onnx
│   └── Reverb_HQ_By_FoxJoy.onnx
├── rvc_models/
│   ├── list.json                 # Curated voice model list (gallery source)
│   ├── public_models.json        # Public index for search/tag filtering
│   ├── MODELS.txt
│   ├── Klee/                     # Example downloaded voice model
│   │   ├── Klee.pth
│   │   └── Klee.index
│   └── ...
├── images/
│   ├── default_model.png         # Fallback image for models without a URL
│   ├── models/                   # Cached model images (auto-downloaded)
│   │   ├── Klee.jpg
│   │   ├── Emilia.jpg
│   │   └── ...
│   ├── webui_generate.png
│   ├── webui_dl_model.png
│   └── webui_upload_model.png
├── src/
│   ├── webui.py                  # Gradio WebUI
│   ├── download_models.py        # JSON-based model downloader (CLI + WebUI)
│   ├── main.py                   # CLI entry point
│   └── ...
├── song_output/                  # Generated covers
└── requirements.txt
```

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
