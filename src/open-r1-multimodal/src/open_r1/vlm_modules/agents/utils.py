import sys
import os
#current_file = os.path.abspath(__file__)
#parent_dir = os.path.dirname(os.path.dirname(current_file))

#sys.path.append(parent_dir)
import json
from pathlib import Path
from .table_utils.tar_to_tex import get_table_tex
from .table_utils.tex_to_image import generate_image
from .table_utils.process_path import process_path
import base64
import numpy as np
from openai import OpenAI

from PIL import Image, ImageOps
from matplotlib import colors

import re
import dotenv
dotenv.load_dotenv()


def extract_latex_code(text: str) -> str:
    """
    Extracts LaTeX code from a markdown code block, with or without a language label.
    Accepts ```latex, ```Latex, or even unlabeled triple-backtick code blocks.

    Args:
        text (str): The full input text containing the code block.

    Returns:
        str: The extracted code content, or an empty string if no match is found.
    """
    pattern = r"```[ \t]*(?i:latex)?[ \t]*\n([\s\S]*?)\n```"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""



def run_code_generate(code, output_name):

    if "plt.show()" in code:
        code = code.replace("plt.show()", "")
    if "plt.tight_layout()" not in code:
        code += "\nplt.tight_layout()"
    if "plt.savefig" not in code:
        code += f"\nplt.savefig('{output_name}.png')\nplt.savefig('{output_name}.pdf')"
    else:
        lines_to_be_deleted = []
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "plt.savefig" in line:
                lines_to_be_deleted.append(i)
        lines = [line for i, line in enumerate(lines) if i not in lines_to_be_deleted]
        lines.append(f"plt.savefig('{output_name}.png')\nplt.savefig('{output_name}.pdf')")
        code = "\n".join(lines)
        
    with open(f"{output_name}.py", "w") as f:
        f.write(code)
    
    code = (
    "try:\n"
    + "\n".join("    " + line for line in code.strip().splitlines())
    + "\nexcept Exception as e:\n"
    + "    import matplotlib.pyplot as plt\n"
    + "    plt.savefig('{output_name}.pdf')\n".format(output_name=output_name)
    + "    plt.savefig('{output_name}.png')\n".format(output_name=output_name)
    + "    with open('{output_name}_error.txt', 'w') as f:\n".format(output_name=output_name)
    + "        f.write(str(e))\n"
    + "    pass\n"
    )

    with open(f"{output_name}_evaluate.py", "w") as f:
        f.write(code)
    
    try:
        os.system(f"python3 {output_name}_evaluate.py")
    except Exception as e:
        import matplotlib.pyplot as plt
        plt.savefig(f"{output_name}.pdf")
        plt.savefig(f"{output_name}.png")
        with open(f"{output_name}_error.txt", "w") as f:
            f.write(str(e))
        pass
    # os.system(f"rm {output_name}_evaluate.py")

def extract_latex_table(latex_code):
    """
    提取 LaTeX 字符串中第一个 \begin{table} 与 \end{table} 之间的内容。
    """
    pattern = r"\\begin{table}.*?\\end{table}"
    match = re.search(pattern, latex_code, re.DOTALL)
    if match is not None:
        return match.group(0)
    else:
        print("Warning: No LaTeX table found in prediction.")
        return None

def extract_validate_run_code(code, output_name):
    try:
        #写入tex文件
        output_file = f"{output_name}.tex"
        # 调用 get_table_tex
        success = get_table_tex(code, str(output_file))
        if success is False:
            print(f"Fail to get table tex: {output_file}.")
            return False
        #编译图片到{output_name}.png
        if not generate_image(f"{output_name}.tex"):
            return False
        return True
    except:
        print(f"Fail to get table tex: {output_file}.")
        return False

def combine_images(image_path1, image_path2, output_path, border_size=10, border_color=(0, 0, 0)):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    if img1.height != img2.height:
        new_height = min(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * new_height / img1.height), new_height))
        img2 = img2.resize((int(img2.width * new_height / img2.height), new_height))
    
    img1_with_border = ImageOps.expand(img1, border=border_size, fill=border_color)
    img2_with_border = ImageOps.expand(img2, border=border_size, fill=border_color)
    
    combined_width = img1_with_border.width + img2_with_border.width
    max_height = max(img1_with_border.height, img2_with_border.height)
    
    new_image = Image.new('RGB', (combined_width, max_height), color=(255, 255, 255))

    new_image.paste(img1_with_border, (0, 0))
    new_image.paste(img2_with_border, (img1_with_border.width, 0))
    new_image.save(output_path)

    
from PIL import Image
from matplotlib import colors
import matplotlib.colors as mcolors

def dominant_colors_extractor(combine_images_path, num_colors=20):
    image = Image.open(combine_images_path)
    
    width, height = image.size
    left_image = image.crop((0, 0, width // 2, height))
    right_image = image.crop((width // 2, 0, width, height))
    images = [left_image, right_image]
    
    left_colors, right_colors = [], []
    for i in range(2):
        image = images[i]
        image = image.resize((100, 100))
        image_data = image.getdata()
        
        color_count = {}
        for pixel in image_data:
            rgb = tuple(pixel[:3])
            color_count[rgb] = color_count.get(rgb, 0) + 1
        
        sorted_colors = sorted(color_count.items(), key=lambda item: item[1], reverse=True)
        dominant_colors = sorted_colors[:num_colors]
        hex_colors = [mcolors.rgb2hex([c / 255.0 for c in color[0]]) for color in dominant_colors]
        if i == 0:
            left_colors = [color.upper() for color in hex_colors if color != "#FFFFFF"]
        else:
            right_colors = [color.upper() for color in hex_colors if color != "#FFFFFF"]
            
    return left_colors, right_colors