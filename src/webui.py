import json
import os
import shutil
import sys
import urllib.request
import zipfile
from argparse import ArgumentParser
from io import StringIO

import gradio as gr

from main import song_cover_pipeline
import download_models as dl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)


def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if not model['name'] in voice_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)

    tags = list(public_models['tags'].keys())
    return gr.update(value=models_table), gr.CheckboxGroup.update(choices=tags)


def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    # move model and index file to extraction folder
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # remove any unnecessary nested folders
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc='[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully downloaded!'

    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully uploaded!'

    except Exception as e:
        raise gr.Error(str(e))


def filter_models(tags, query):
    models_table = []

    # no filter
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on tags and query
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on only tags
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on only query
    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    return gr.update(value=models_table)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.update(value=pub_models.loc[event.index[0], 'URL']), gr.Text.update(value=pub_models.loc[event.index[0], 'Model Name'])


def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


# ── JSON Index tab helpers ────────────────────────────────────────────

def get_model_image_path(model_name):
    """Return the local image path for a model (downloads if needed)."""
    default_img = os.path.join(BASE_DIR, 'images', 'default_model.png')
    local_images_dir = os.path.join(BASE_DIR, 'images', 'models')

    # Check for locally cached image
    local_path = os.path.join(local_images_dir, f"{model_name}.jpg")
    if os.path.exists(local_path):
        return local_path

    # Try to download from URL
    model = dl.get_voice_model(model_name)
    if model and model.get('image', '').startswith('http'):
        try:
            os.makedirs(local_images_dir, exist_ok=True)
            urllib.request.urlretrieve(model['image'], local_path)
            return local_path
        except Exception:
            pass

    return default_img if os.path.exists(default_img) else None


def on_json_model_select(model_name):
    """When a model is selected, update image + info."""
    model = dl.get_voice_model(model_name)
    if not model:
        return gr.update(value=None), gr.update(value="Select a model...")

    image_url = get_model_image_path(model_name)
    info_text = (
        f"**{model['name']}**\n\n"
        f"{model.get('description', 'No description')}\n\n"
        f"Credit: {model.get('credit', 'N/A')}\n\n"
        f"URL: `{model.get('url', '')}`"
    )
    downloaded = os.path.exists(os.path.join(rvc_models_dir, model_name))
    if downloaded:
        info_text += "\n\n✅ **Already downloaded**"
    else:
        info_text += "\n\n⬇️ Click **Download** to install"

    return gr.update(value=image_url), gr.update(value=info_text)


def download_json_voice_model(model_name, progress=gr.Progress()):
    """Download a voice model from rvc_models/list.json."""
    try:
        result = dl.download_voice_model(model_name, progress_callback=lambda m: progress(0.5, desc=m))
        # Refresh info after download
        model = dl.get_voice_model(model_name)
        image_url = get_model_image_path(model_name)
        info_text = (
            f"**{model['name']}**\n\n"
            f"{model.get('description', '')}\n\n"
            f"Credit: {model.get('credit', 'N/A')}\n\n"
            f"✅ **Downloaded!**\n\n{result}"
        )
        return result, gr.update(value=image_url), gr.update(value=info_text)
    except Exception as e:
        raise gr.Error(str(e))


def download_all_required_models(progress=gr.Progress()):
    """Download hubert, rmvpe, and MDX-Net models from models_manifest.json."""
    try:
        ok = dl.download_required()
        return '[+] All required models downloaded!' if ok else '[ERROR] Some downloads failed. Check console.'
    except Exception as e:
        raise gr.Error(str(e))


def check_model_status():
    """Check which models are present."""
    try:
        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        dl.check_status()
        sys.stdout = old_stdout
        return buf.getvalue()
    except Exception as e:
        raise gr.Error(str(e))


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    # Load voice model list from rvc_models/list.json
    json_voice_models = dl.get_voice_list()
    json_voice_names = sorted([m['name'] for m in json_voice_models])

    # Build gallery of model images
    # Gradio 6.x: Gallery value must be list of (image_path_or_url, caption) tuples
    default_img = os.path.join(BASE_DIR, 'images', 'default_model.png')
    local_images_dir = os.path.join(BASE_DIR, 'images', 'models')
    os.makedirs(local_images_dir, exist_ok=True)

    gallery_value = []
    for m in json_voice_models:
        img = m.get('image', '')
        image_src = None

        if img and img.startswith('http'):
            # Download remote images locally to avoid hotlink protection issues
            local_path = os.path.join(local_images_dir, f"{m['name']}.jpg")
            if not os.path.exists(local_path):
                try:
                    urllib.request.urlretrieve(img, local_path)
                except Exception:
                    if os.path.exists(default_img):
                        local_path = default_img
                    else:
                        continue  # skip if no image available
            image_src = local_path
        elif os.path.exists(default_img):
            image_src = default_img
        else:
            continue  # skip models with no image

        if image_src is None:
            continue

        status = "✅" if os.path.exists(os.path.join(rvc_models_dir, m['name'])) else "⬇️"
        gallery_value.append((image_src, f"{status} {m['name']}"))

    with gr.Blocks(title='AICoverGenWebUI') as app:

        gr.Label('AICoverGen WebUI created with ❤️', show_label=False)

        # main tab
        with gr.Tab("Generate"):

            with gr.Accordion('Main Options'):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Models folder "AICoverGen --> rvc_models". After new models are added into this folder, click the refresh button')
                        ref_btn = gr.Button('Refresh Models 🔁', variant='primary')

                    with gr.Column() as yt_link_col:
                        song_input = gr.Text(label='Song input', info='Link to a song on YouTube or full path to a local file. For file upload, click the button below.')
                        show_file_upload_button = gr.Button('Upload file instead')

                    with gr.Column(visible=False) as file_upload_col:
                        local_file = gr.File(label='Audio file')
                        song_input_file = gr.UploadButton('Upload 📂', file_types=['audio'], variant='primary')
                        show_yt_link_button = gr.Button('Paste YouTube link/Path to local file instead')
                        song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                    with gr.Column():
                        pitch = gr.Slider(-3, 3, value=0, step=1, label='Pitch Change (Vocals ONLY)', info='Generally, use 1 for male to female conversions and -1 for vice-versa. (Octaves)')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Overall Pitch Change', info='Changes pitch/key of vocals and instrumentals together. Altering this slightly reduces sound quality. (Semitones)')
                    show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

            with gr.Accordion('Voice conversion options', open=False):
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much of the AI voice's accent to keep in the vocals")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter radius', info='If >=3: apply median filtering median filtering to the harvested pitch results. Can reduce breathiness')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS mix rate', info="Control how much to mimic the original vocal's loudness (0) or a fixed loudness (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Protect rate', info='Protect voiceless consonants and breath sounds. Set to 0.5 to disable.')
                    with gr.Column():
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Pitch detection algorithm', info='Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals)')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe hop length', info='Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy.')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                keep_files = gr.Checkbox(label='Keep intermediate files', info='Keep all audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals. Leave unchecked to save space')

            with gr.Accordion('Audio mixing options', open=False):
                gr.Markdown('### Volume Change (decibels)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocals')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocals')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music')

                gr.Markdown('### Reverb Control on AI Vocals')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room size', info='The larger the room, the longer the reverb time')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Wetness level', info='Level of AI vocals with reverb')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Dryness level', info='Level of AI vocals without reverb')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping level', info='Absorption of high frequencies in the reverb')

                gr.Markdown('### Audio Output Format')
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Output file type', info='mp3: small file size, decent quality. wav: Large file size, best quality')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Clear', components=[song_input, rvc_model, keep_files, local_file])
                generate_btn = gr.Button("Generate", variant='primary')
                ai_cover = gr.Audio(label='AI Cover', show_share_button=False)

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                               inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                       inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                       protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                                       output_format],
                               outputs=[ai_cover])
            clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
                            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                                     reverb_dry, reverb_damping, output_format, ai_cover])

        # Download tab
        with gr.Tab('Download model'):

            with gr.Tab('From JSON Index'):
                gr.Markdown('## 📋 Voice Model Gallery')
                gr.Markdown('Select a model from the dropdown or browse the gallery. Click **Download** to install.')

                # Gallery of all model images
                model_gallery = gr.Gallery(
                    value=gallery_value,
                    label="Model Gallery",
                    columns=6,
                    rows=10,
                    height='auto',
                    object_fit='cover',
                    allow_preview=False,
                    show_label=False,
                    selected_index=None,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        json_model_select = gr.Dropdown(
                            choices=json_voice_names,
                            label='Voice Model',
                            info='Choose a model to download'
                        )
                    with gr.Column(scale=1):
                        json_download_btn = gr.Button('Download Model', variant='primary')
                    with gr.Column(scale=1):
                        dl_required_btn = gr.Button('Download Required Models', variant='secondary')
                    with gr.Column(scale=1):
                        check_status_btn = gr.Button('Check Model Status')

                # Model preview: image + info
                with gr.Row():
                    with gr.Column(scale=1):
                        model_preview_image = gr.Image(
                            label='Model Preview',
                            height=300,
                            interactive=False,
                            show_label=True,
                        )
                    with gr.Column(scale=2):
                        model_preview_info = gr.Markdown('Select a model to see details...')

                json_dl_output = gr.Text(label='Download Status', interactive=False)

                # Events
                json_model_select.change(
                    on_json_model_select,
                    inputs=[json_model_select],
                    outputs=[model_preview_image, model_preview_info]
                )

                json_download_btn.click(
                    download_json_voice_model,
                    inputs=[json_model_select],
                    outputs=[json_dl_output, model_preview_image, model_preview_info]
                )

                dl_required_btn.click(
                    download_all_required_models,
                    outputs=[json_dl_output]
                )
                check_status_btn.click(
                    check_model_status,
                    outputs=[json_dl_output]
                )

            with gr.Tab('From HuggingFace/Pixeldrain URL'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Download link to model', info='Should be a zip file containing a .pth model file and an optional .index file.')
                    model_name = gr.Text(label='Name your model', info='Give your new model a unique name from your other voice models.')

                with gr.Row():
                    download_btn = gr.Button('Download', variant='primary', scale=19)
                    dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('## Input Examples')
                gr.Examples(
                    [
                        ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                        ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                        ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']
                    ],
                    [model_zip_link, model_name],
                    [],
                    download_online_model,
                )

            with gr.Tab('From Public Index'):

                gr.Markdown('## How to use')
                gr.Markdown('- Click Initialize public models table')
                gr.Markdown('- Filter models using tags or search bar')
                gr.Markdown('- Select a row to autofill the download link and model name')
                gr.Markdown('- Click Download')

                with gr.Row():
                    pub_zip_link = gr.Text(label='Download link to model')
                    pub_model_name = gr.Text(label='Model name')

                with gr.Row():
                    download_pub_btn = gr.Button('Download', variant='primary', scale=19)
                    pub_dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)

                filter_tags = gr.CheckboxGroup(value=[], label='Show voice models with tags', choices=[])
                search_query = gr.Text(label='Search')
                load_public_models_button = gr.Button(value='Initialize public models table', variant='primary')

                public_models_table = gr.DataFrame(value=[], headers=['Model Name', 'Description', 'Credit', 'URL', 'Tags'], label='Available Public Models', interactive=False)
                public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

        # Upload tab
        with gr.Tab('Upload model'):
            gr.Markdown('## Upload locally trained RVC v2 model and index file')
            gr.Markdown('- Find model file (weights folder) and optional index file (logs/[name] folder)')
            gr.Markdown('- Compress files into zip file')
            gr.Markdown('- Upload zip file and give unique name for voice')
            gr.Markdown('- Click Upload model')

            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(label='Zip file')

                local_model_name = gr.Text(label='Model name')

            with gr.Row():
                model_upload_button = gr.Button('Upload model', variant='primary', scale=19)
                local_upload_output_message = gr.Text(label='Output Message', interactive=False, scale=20)
                model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
