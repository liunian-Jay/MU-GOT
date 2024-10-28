import os
import re
import torch
import string
import argparse
import requests
from PIL import Image
from io import BytesIO

from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria


from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import KeywordsStoppingCriteria
from vllm.sampling_params import SamplingParams

# 从本地加载模型
from PDF_parsing import tokenizer, model, image_processor, image_processor_high

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def load_image(image_bytes):
     # 从bytes加载图像并转换为RGB格式
    if isinstance(image_bytes, bytes):
        try:       
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    # 如果是文件或者下载链接，用于测试
    image_file = image_bytes
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def old_batch_got(image_bytes, type='format'):
    # 使用修改后的load_image函数来加载图像
    images = []
    for image_bytes in image_list:
        image = load_image(image_bytes)
        if image is None:
            return "Image loading failed."
        images.append(image)


    # 构建提示符
    qs = f'OCR with format: ' if type == 'format' else 'OCR: '

    use_im_start_end = True
    image_token_len = 256

    if use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_PATCH_TOKEN * image_token_len}{DEFAULT_IM_END_TOKEN}\n{qs}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

    # 配置对话模板
    conv_mode = "mpt"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensors = []
    image_tensors_1 = []
    for image in images:
        image_tensor = image_processor(image)
        image_tensor_1 = image_processor_high(image.copy())
        image_tensors.append(image_tensor)
        image_tensors_1.append(image_tensor_1)

    # 修改by jianBo
    inputs = tokenizer([prompt]*len(image_list))
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # 修改 By YiJiang
    tokenizer.eos_token_id = tokenizer.pad_token_id
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda()) for image_tensor, image_tensor_1 in zip(image_tensors, image_tensors_1)],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria],

            eos_token_id=tokenizer.eos_token_id,  # 添加结束标记
            pad_token_id=tokenizer.pad_token_id,  # 添加填充标记
        )
        outputs = []
        for i in range(len(output_ids)):
            output = tokenizer.decode(output_ids[i, input_ids.shape[1]:], skip_special_tokens=True).strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            outputs.append(output)
    return outputs

def vllm_got(image_list, type='format'):
    # 使用修改后的load_image函数来加载图像
    images = []
    for image_bytes in image_list:
        image = load_image(image_bytes)
        if image is None:
            return "Image loading failed."
        images.append(image)
    
    image_tensors = []
    image_tensors_1 = []
    for image in images:
        image_tensor = image_processor(image)
        image_tensor_1 = image_processor_high(image.copy())
        image_tensors.append(image_tensor)
        image_tensors_1.append(image_tensor_1)

    # 构建提示符
    qs = f'OCR with format: ' if type == 'format' else 'OCR: '
    use_im_start_end = True
    image_token_len = 256

    if use_im_start_end:
        qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_PATCH_TOKEN * image_token_len}{DEFAULT_IM_END_TOKEN}\n{qs}"
    else:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

    # 配置对话模板
    conv_mode = "mpt"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer(prompt)


    input_ids = inputs.input_ids
    tokenizer.eos_token_id = tokenizer.pad_token_id
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
    new_keyword_ids = [keyword_id[0] for keyword_id in keyword_ids]
    sampling_param = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.25, max_tokens=2048, stop_token_ids = new_keyword_ids)

    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            prompts = [
                {
                    'prompt_token_ids': input_ids,
                    'multi_modal_data':{
                        'image':image_tensor.unsqueeze(0)
                    }
                } for image_tensor in image_tensors
            ],
            sampling_params = sampling_param
        )
        generated_text = ""
        for o in output_ids:
            print('-'*100,'\n\n')
            generated_text += o.outputs[0].text

    return generated_text



# if __name__ == '__main__':
#     import time
#     start = time.time()
#     image_list = [
#         'xxxxxx.jpg'
#     ]          
#     res = vllm_got(image_list)
#     print()
#     end = time.time()
#     print(f"Time cost: {end-start:.3f} seconds")
