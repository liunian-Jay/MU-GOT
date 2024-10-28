import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


 
translation_table = str.maketrans(punctuation_dict)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(image_file):
    model_name="/home/brhu/GOT-OCR2.0/GOT-OCR-2.0-master/GOT_weights/GOT_weights"
    type='format'
    box=''
    color=''
    render=False
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    model.to(device='cuda', dtype=torch.bfloat16)

    # TODO vary old codes, NEED del 
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    use_im_start_end = True
    image_token_len = 256

    image = load_image(image_file)
    w, h = image.size
    
    if type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if box:
        bbox = eval(box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0] / w * 1000)
            bbox[1] = int(bbox[1] / h * 1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0] / w * 1000)
            bbox[1] = int(bbox[1] / h * 1000)
            bbox[2] = int(bbox[2] / w * 1000)
            bbox[3] = int(bbox[3] / h * 1000)
        if type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if color:
        if type == 'format':
            qs = '[' + color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"
    conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)

    inputs = tokenizer([prompt])
    image_1 = image.copy()
    image_tensor = image_processor(image)
    image_tensor_1 = image_processor_high(image_1)

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
        )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # Render if needed
        if render:
            print('==============rendering===============')
            # ... (Rendering logic)

        # Return the final output text
        return outputs

def process_markdown(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    image_pattern = re.compile(r'!\[.*?\]\(([^)]+\.jpg)\)')
    image_paths = image_pattern.findall(md_content)

    updated_content = md_content

    for image_path in image_paths:
        try:
            ocr_result = eval_model(image_path)
            ocr_text = f'\n<!-- OCR Result for {image_path} -->\n{ocr_result}\n'
            pattern = r'!\[[^\]]*\]\(' + re.escape(image_path) + r'\)'
            target_string = re.findall(pattern, md_content)[0]
            updated_content = updated_content.replace(target_string, ocr_text)
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")

    try:
        with open(md_file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        print(f"Markdown 文件 {md_file_path} 已成功更新！")
    except Exception as e:
        print(f"写入文件时出错: {e}")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-name", type=str, default="/home/brhu/GOT-OCR2.0/GOT-OCR-2.0-master/GOT_weights/GOT_weights")
    # parser.add_argument("--image-file", type=str, default="/home/brhu/GOT-OCR2.0/GOT-OCR-2.0-master/images/b320c6757cc91828012e515616fd2de455e0fbd00f7599c37b24ba4311521020.jpg")
    # parser.add_argument("--type", type=str, required=True)
    # parser.add_argument("--box", type=str, default= '')
    # parser.add_argument("--color", type=str, default= '')
    # parser.add_argument("--render", action='store_true')
    # args = parser.parse_args()

    # res = eval_model(image_file="/home/brhu/GOT-OCR2.0/GOT-OCR-2.0-master/images/b320c6757cc91828012e515616fd2de455e0fbd00f7599c37b24ba4311521020.jpg")
    # print(res)

    process_markdown(md_file_path = "/home/brhu/GOT-OCR2.0/GOT-OCR-2.0-master/MinerU_outpdf/hulu.md")