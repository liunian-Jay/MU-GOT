from PDF_parsing import pdf2md, vllm_got, process_markdown


def pdf2md_api(doc_file) ->tuple[str,dict[str,bytes]]:
    """ MinerU识别 """
    md_content, images_and_path = pdf2md(doc_file)
    return md_content, images_and_path


def pic2table_api(image_bytes: bytes, type: str = 'format') -> str:
    """ GOT模块 """
    result = vllm_got(image_list=[image_bytes], type=type)
    return result


def pdf2txt_api(doc_file):
    # 初步解析
    md_content, images_and_path = pdf2md_api(doc_file)
    # 调用 pic2table_api 进行表格识别
    new_imgs_and_path = dict()
    for img_path, img_data in images_and_path.items():
        if isinstance(img_data, bytes):
            ocr_result = pic2table_api(img_data)
            new_imgs_and_path[img_path] = ocr_result
        else:
            print(f"图片 {img_path} 的数据不是字节类型，无法处理。")
    # txt进行处理
    txt = process_markdown(md_content, new_imgs_and_path)
    return txt


if __name__ == '__main__':
    pdf = 'xxxx'
    """输入的参数可以是pdf_path or file-like object"""
    txt = pdf2txt_api(pdf)