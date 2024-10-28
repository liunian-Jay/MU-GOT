import os
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.pipe.TXTPipe import TXTPipe
import magic_pdf.model as model_config 
model_config.__use_inside_model__ = True

def read_fn(path):
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

def pdf2md(file):
    if not isinstance(file, str):
        # If 'file' is not a string, assume it's a file-like object and process it
        pdf_data = read_fn(file)
    else:
        # If 'file' is a string (likely a file path), open it and process
        with open(file, 'rb') as f:  # 'rb' mode to read binary files like PDFs
            pdf_data = f.read()

    image_writer = DiskReaderWriter('')
    image_dir = ''

    pipe = TXTPipe(pdf_data, [], image_writer)
    pipe.pipe_analyze()
    pipe.pipe_parse()

    images_and_path = {}
    for page_info in pipe.pdf_mid_data['pdf_info']:
        if 'images_and_path' in page_info:
            for item in page_info['images_and_path']:
                if 'image_path' in item and 'byte_data' in item:
                    images_and_path[item['image_path']] = item['byte_data']
        page_info.pop('images_and_path')
        
    md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
    return md_content, images_and_path
