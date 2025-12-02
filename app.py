import argparse
import math
import json
import os
import os.path as osp
from pathlib import Path
from shutil import copyfile
import sys
import random
import yaml

import cv2
import gradio as gr
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'eval')))
from eval.t3_dataset import draw_glyph2
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll

ASPECT_RATIO_LD_LIST = [  # width:height
    "2.39:1",  # cinemascope, 2.39
    "2:1",  # rare, 2
    "16:9",  # rare, 1.89
    "1.85:1",  # american widescreen, 1.85
    "9:16",  # popular, 1.78
    "5:8",  # rare, 1.6
    "3:2",  # rare, 1.5
    "4:3",  # classic, 1.33
    "1:1",  # square
]

RESOLUTIONS = [512, 768, 1024]
PIXELS = [512 * 512, 768 * 768, 1024 * 1024]

def get_ratio(name: str) -> float:
    width, height = map(float, name.split(":"))
    return height / width

def get_closest_ratio(height: float, width: float, ratios: dict) -> str:
    aspect_ratio = height / width
    closest_ratio = min(
        ratios, key=lambda ratio: abs(aspect_ratio - get_ratio(ratio))
    )
    return closest_ratio

def get_aspect_ratios_dict(
    total_pixels: int = 256 * 256, training: bool = True
) -> dict[str, tuple[int, int]]:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    aspect_ratios_dict = {}
    aspect_ratios_vertical_dict = {}
    for ratio in ASPECT_RATIO_LD_LIST:
        width_ratio, height_ratio = map(float, ratio.split(":"))
        width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
        height = int((total_pixels / width) // D) * D

        if training:
            # adjust aspect ratio to match total pixels
            diff = abs(height * width - total_pixels)
            candidate = [
                (height - D, width),
                (height + D, width),
                (height, width - D),
                (height, width + D),
            ]
            for h, w in candidate:
                if abs(h * w - total_pixels) < diff:
                    height, width = h, w
                    diff = abs(h * w - total_pixels)

        # remove duplicated aspect ratio
        if (height, width) not in aspect_ratios_dict.values() or not training:
            aspect_ratios_dict[ratio] = (height, width)
            vertial_ratios = ":".join(ratio.split(":")[::-1])
            aspect_ratios_vertical_dict[vertial_ratios] = (width, height)

    aspect_ratios_dict.update(aspect_ratios_vertical_dict)

    return aspect_ratios_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default='models/anytext_v1.1.ckpt',
        help='path of model'
    )
    parser.add_argument(
        '--config_path',
        type=str,

    )
    args = parser.parse_args()
    return args

def init_pipeline(args, config):
    training_config = config["train"]
    trainable_model = OminiModelFIll(
            flux_pipe_id=config["flux_path"],
            lora_config=training_config["lora_config"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=config.get("model", {}),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )

    from safetensors.torch import load_file
    state_dict = load_file(args.model_path)
    state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
    trainable_model.transformer.load_state_dict(state_dict1, strict=False)

    pipe = trainable_model.flux_pipe
    
    return pipe, trainable_model

def get_captions(ori_image, _input_file):
    image = Image.fromarray(ori_image)
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = blipmodel.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    caption = f'{generated_text}, that reads "{_input_file}"'
    return caption

def get_glyph_pos(mask, _input_file, width, height, add_space=False):
    mask = mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hint = mask / 255
    glyph_scale = 1

    # Parse multiple texts: split by newlines
    if isinstance(_input_file, str):
        text_list = [t.strip() for t in _input_file.splitlines() if t.strip()]
    else:
        text_list = []

    # Sort contours from top to bottom using top-most y
    def contour_top_y(cnt):
        ys = cnt[:, 0, 1] if len(cnt.shape) == 3 else cnt[:, 1]
        return int(np.min(ys)) if ys.size > 0 else 0

    contours = sorted(contours, key=contour_top_y)

    # Compose glyphs by placing each text into the corresponding contour
    glyphs_canvas = np.zeros((height, width, 1), dtype=np.float64)
    for idx, cnt in enumerate(contours):
        if idx >= len(text_list):
            break
        gly = draw_glyph2(selffont, text_list[idx], cnt, scale=glyph_scale, width=width, height=height, add_space=add_space)
        glyphs_canvas = np.maximum(glyphs_canvas, gly)

    return hint, glyphs_canvas

def brush_button_func(brush_image):
    _mask = brush_image['layers'][0][:, :, :3]
    _mask = np.where(_mask > 0, 255, 0)
    return _mask[:,:,0], [_mask[:,:,0]]

def update_mask_func(edit_mask, edit_text, add_space):
    background = edit_mask['background']
    background = background[:, :, :3]
    mask, _ = brush_button_func(edit_mask)

    hint, glyphs = get_glyph_pos(mask, edit_text, background.shape[1], background.shape[0], add_space=add_space)
    hint = hint.astype('uint8') * 255
    glyphs = (1 - glyphs.astype('uint8') ) * 255
    glyphs = glyphs[:,:,0]
    caption = get_captions(background, edit_text)
    return hint, glyphs, caption

def generate_image_func(prompt, img, glyph_img, mask_img, seed):
    ori_width, ori_height = img.size
    num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
    aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
    close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
    tgt_height, tgt_width = aspect_ratio_dict[close_ratio]
    
    hint = mask_img.resize((tgt_width, tgt_height)).convert('RGB')
    img = img.resize((tgt_width, tgt_height))
    condition_img = glyph_img.resize((tgt_width, tgt_height)).convert('RGB')
    hint = np.array(hint) / 255
    condition_img = np.array(condition_img)
    condition_img = (255 - condition_img) / 255
    condition_img = [condition_img, hint, img]
    position_delta = [0, 0]
    condition = Condition(
                    condition_type='word_fill',
                    condition=condition_img,
                    position_delta=position_delta,
                )
    generator.manual_seed(seed)
    res = generate_fill(
        pipe,
        prompt=prompt,
        conditions=[condition],
        height=tgt_height,
        width=tgt_width,
        generator=generator,
        model_config=config.get("model", {}),
        default_lora=True,
    )
    return res.images[0]

def mode_select_change_func(mode_select):
    if mode_select:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def image_upload_func(image):
    return gr.update(value=image)

single_examples = [
    ["lepto college of education, the written materials on the picture: LESOTHO , COLLEGE OF , RE BONA LESELI LESEL , EDUCATION .", Image.open("assets/hint_imgs.jpg"), Image.open("assets/hint_imgs_word.png"), Image.open("assets/hint.png"), 42],
    ["keda group logo, that reads KDG , 科达股份 , 证券代码：600986 , 数字营销领军集团 .", Image.open("assets/hint_imgs1.jpg"), Image.open("assets/hint_imgs_word1.png"), Image.open("assets/hint1.png"), 42],
    ["chinese calligraphy font with the word 'love' written in it, that reads 精神食粮 .", Image.open("assets/hint_imgs2.jpg"), Image.open("assets/hint_imgs_word2.png"), Image.open("assets/hint2.png"), 42],
]
# Create the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with FLUX-Text")
    gr.Markdown("Generate images using FLUX-Text with a lightweight Condition Injection LoRA.")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                img = gr.Image(label="Image", type="pil")  # 上传图像文件
            with gr.Row():
                glyph_img = gr.Image(label="glyph Image", type="pil")  # 上传图像文件
                mask_img = gr.Image(label="mask Image", type="pil")  # 上传图像文件

            with gr.Row():
                mode_select = gr.Checkbox(label='manual edit', info='manual checkbox')

            with gr.Row(visible=False) as edit_image:
                with gr.Column():
                    edit_mask = gr.ImageEditor(type="numpy", interactive=True)
                with gr.Column():
                    edit_text = gr.Textbox(label="Input texts", lines=4, placeholder="Enter one text per line (top to bottom)")
                    add_space = gr.Checkbox(label="Add spacing to fill mask", value=False)
                    edit_btn = gr.Button("Generate glyph and mask")

            prompt = gr.Textbox(label="Prompt")
            seed = gr.Number(label="Seed", value=42)
            single_generate_btn = gr.Button("Generate Image")
        with gr.Column():
            single_output_image = gr.Image(label="Generated Image")

    # Add examples for Single Condition Generation
    gr.Examples(
        examples=single_examples,
        inputs=[prompt, img, glyph_img, mask_img, seed],
        outputs=single_output_image,
        fn=generate_image_func,
        cache_examples=False,  # 缓存示例结果以加快加载速度
        label="Examples"
    )

    # Link the buttons to the functions
    img.upload(
        image_upload_func,
        inputs=[img],
        outputs=[edit_mask],
    )
    single_generate_btn.click(
        generate_image_func,
        inputs=[prompt, img, glyph_img, mask_img, seed],
        outputs=single_output_image
    )
    mode_select.change(
        mode_select_change_func,
        inputs=[mode_select],
        outputs=[edit_image],
    )
    edit_btn.click(
        update_mask_func,
        inputs=[edit_mask, edit_text, add_space],
        outputs=[mask_img, glyph_img, prompt],
    )

if __name__ == '__main__':
    font_path='./font/Arial_Unicode.ttf'
    selffont = ImageFont.truetype(font_path, size=60)
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    generator = torch.Generator(device="cuda")
    to_tensor = T.ToTensor()

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipmodel = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blipmodel.to(device, torch.float16)

    pipe, trainable_model = init_pipeline(args, config)

    # Launch the Gradio app
    demo.queue().launch(server_name="0.0.0.0", server_port=6681, share=False)