# coding=utf-8
import os
import subprocess
from tqdm import tqdm
from pathlib import Path

def generate_image(tex_path):
    pdf_path = tex_path.replace(".tex", ".pdf").replace("\\", "/")
    img_path = tex_path.replace(".tex", "")
    aux_path = pdf_path.replace(".pdf", ".aux")
    log_path = pdf_path.replace(".pdf", ".log")
    out_path = pdf_path.replace(".pdf", ".out")
    
    try:
        # 1. 运行 pdflatex 转换为 PDF
        folder_path = str(Path(tex_path).parent)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path], 
                        cwd=folder_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except:
        print(f"ERROR 1 pdflatex fail): {tex_path}")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(aux_path):
            os.remove(aux_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        return False
    try:   
        # 2. 运行 pdfcrop 裁剪边缘，保留 5pt 边距
        subprocess.run(["pdfcrop","--margins", "5 5 5 5", pdf_path, pdf_path], #"--margins", "\"5 5 5 5\""
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"ERROR 2 pdfcrop fail: {tex_path}")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(aux_path):
            os.remove(aux_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        return False
    try:   
        # 3. 运行 pdftoppm 转换为 PNG
        subprocess.run(["pdftoppm", "-png", "-r", "300", pdf_path, img_path], 
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        output_img = img_path + "-1.png"         # 生成的实际文件
        final_img = img_path + ".png"
        if os.path.exists(output_img):
            os.rename(output_img, final_img)
    except subprocess.CalledProcessError:
        print(f"ERROR 3 pdftoppm fail: {tex_path}")
        return False
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(aux_path):
        os.remove(aux_path)
    if os.path.exists(log_path):
        os.remove(log_path)
    if os.path.exists(out_path):
        os.remove(out_path)
    return True