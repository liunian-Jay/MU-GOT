import re
import os



def process_markdown(md_content, images_and_path):
    # 匹配图片的正则表达式，支持常见图片格式
    image_pattern = re.compile(r'!\[.*?\]\(([^)]+\.(?:jpg|jpeg|png|gif))\)')
    image_paths = image_pattern.findall(md_content)

    if not image_paths:
        print("没有找到任何图片需要处理。")
        return md_content

    updated_content = md_content
    for image_path in image_paths:
        try:
            # 获取字典的 key，中间部分没有前缀和后缀
            key = os.path.splitext(os.path.basename(image_path))[0]  # 仅保留文件名去掉扩展名
            key = key.lstrip('/')
            key = key + ".jpg"
            # 从字典中获取图片数据
            if key in images_and_path:
                ocr_result = images_and_path[key]
                ocr_text = f'\n<!-- OCR Result for {image_path} -->\n{ocr_result}\n'
                pattern = r'!\[[^\]]*\]\(' + re.escape(image_path) + r'\)'
                target_string = re.findall(pattern, md_content)[0]
                updated_content = updated_content.replace(target_string, ocr_text)
            else:
                print(f"找不到图片 {image_path} 的数据。请检查路径是否正确并确保它存在于字典中。")
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
    return updated_content


# updated = process_markdown(md_content, images_and_path)
# print(updated)