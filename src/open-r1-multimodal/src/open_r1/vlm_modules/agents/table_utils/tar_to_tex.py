# coding=utf-8
import os
import tarfile
import shutil
import re

def extract_tar_gz_files(source_dir, output_folder):
    tar_folder = os.path.join(source_dir, "tex")
    total_files = 0  # 记录总的tar.gz文件数
    successful_files = 0  # 记录成功处理的tar.gz文件数
    
    for filename in os.listdir(tar_folder):
        if filename.endswith(".tar.gz"):
            total_files += 1
            tar_path = os.path.join(tar_folder, filename)
            extract_folder = os.path.join(tar_folder, filename.replace(".tar.gz", ""))
            #tex_output_folder = os.path.join(source_dir, "tex_file")
            tex_output_folder = os.path.join(output_folder, f"{filename.replace('.tar.gz', '')}_tex")
            
            # 创建解压缩和tex输出目录
            os.makedirs(extract_folder, exist_ok=True)
            os.makedirs(tex_output_folder, exist_ok=True)
            
            try:
                # 解压tar.gz文件
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(extract_folder)
                
                # 遍历解压出的目录，查找所有.tex文件并移动到tex_output_folder
                for root, _, files in os.walk(extract_folder):
                    for file in files:
                        if file.endswith(".tex"):
                            tex_file_path = os.path.join(root, file)
                            shutil.move(tex_file_path, os.path.join(tex_output_folder, file))
                
                # 如果没有发生异常，计为成功处理
                successful_files += 1
            
            except Exception as e:
                print(f"Error processing {tar_path}: {e}")
            
            finally:
                # 可选择删除解压的中间文件，不论是否成功解压
                if os.path.exists(extract_folder):
                    shutil.rmtree(extract_folder)

    # 输出成功处理的tar文件数和总的tar文件数
    print(f"Total tar.gz files: {total_files}")
    print(f"Successfully processed tar.gz files: {successful_files}")


def remove_commented_sections(tex_content):
    """
    移除 LaTeX 文件中被 % 注释的行
    """
    lines = tex_content.split('\n')
    uncommented_lines = [line for line in lines if not line.strip().startswith('%')]
    return '\n'.join(uncommented_lines)

def remove_latex_references(tex_content):
    """
    移除 \cite、\ref、\citep、\citet、\footnote 等引用命令
    """
    # 匹配命令格式如 \cite{...}，\ref{...}，或带有可选参数的 \citep[xxx]{...}
    reference_patterns =[
    r'\\cite[t|p]?\*?(?:\[[^\]]*\])?\{[^}]*\}',       # \cite, \citep, \citet, \cite*, ...
    r'\\cite(author|year|alp|alt)\*?\{[^}]*\}',       # author/year/alp/alt variants
    r'\\[Cc]ite[a-zA-Z]*\*?\{[^}]*\}',                # 通配大小写Cite命令
    r'\\ref\{[^}]*\}',
    r'\\eqref\{[^}]*\}',
    r'\\autoref\{[^}]*\}',
    r'\\pageref\{[^}]*\}',
    r'\\label\{[^}]*\}',
    r'\\nameref\{[^}]*\}',
    r'\\cref\{[^}]*\}',
    r'\\Cref\{[^}]*\}',
    r'\\crefrange\{[^}]*\}\{[^}]*\}',
    r'\\hyperref(?:\[[^\]]*\])?\{[^}]*\}',
    r'\\footnote\{[^}]*\}',
    r'\\footnotemark',
    r'\\footnotetext\{[^}]*\}'
]
    for pattern in reference_patterns:
        tex_content = re.sub(pattern, '', tex_content)
    tex_content=re.sub(r'\\begin\{table\}\[[^\]]*\]', r'\\begin{table}', tex_content)
    return tex_content

def extract_tables_from_tex(tex_content):
    """
    从 LaTeX 文本中提取所有表格环境的内容
    """
    tex_content = remove_commented_sections(tex_content)  # 移除被注释的内容
    tex_content = remove_latex_references(tex_content)    # 移除引用指令
    table_pattern = re.compile(r'\\begin{table}.*?\\end{table}', re.DOTALL)
    return table_pattern.findall(tex_content)

def analyze_table_for_required_packages(table_content):
    """
    分析表格内容，动态决定所需的 LaTeX 包
    """
    required_packages = set()

    # --- 表格命令和环境 ---
    if re.search(r'\\multirow', table_content):
        required_packages.update(['multirow'])
    if re.search(r'\\toprule|\\midrule|\\bottomrule', table_content):
        required_packages.add('booktabs')
    #if re.search(r'\\hline', table_content):
    #    required_packages.add('array')
    if re.search(r'\\arraystretch', table_content):
        required_packages.add('array')
    # 检查列格式修饰符 >{...} 或 <{...}，来自 array 宏包
    if re.search(r'[><]\s*\{.*?\}', table_content):
        required_packages.add('array')


    # --- 颜色命令 ---
    # 检查基本颜色命令：\textcolor, \color, \definecolor
    if (re.search(r'\\textcolor', table_content) or 
        re.search(r'\\color', table_content) or 
        re.search(r'\\definecolor', table_content)):
        required_packages.add('xcolor')
    # 检查表格内颜色命令：\cellcolor, \rowcolors, \columncolor
    if (re.search(r'\\cellcolor', table_content) or 
        re.search(r'\\rowcolor', table_content) or
        re.search(r'\\columncolor', table_content)):
        required_packages.add('[table]xcolor')
        #required_packages.add('xcolor')
        #required_packages.add('colortbl')
    #if re.search(r'\\cellcolor\{.*?\}', table_content) or re.search(r'\\rowcolors\{.*?\}', table_content):
    # 检查文字高亮命令，如 \hl 或 \st（soul 包支持）
    if re.search(r'\\hl', table_content) or re.search(r'\\st', table_content):
        required_packages.add('soul')

    
    # --- 数学符号与公式 ---
    # 检查常见数学命令，如 \frac, \sqrt, \int, \sum, \prod, \lim, \sin, \cos, \log, \exp 等，
    if re.search(r'\\(frac|sqrt|int|sum|prod|lim|sin|cos|log|exp)\b', table_content):
        required_packages.update(['amsmath'])
    # 检查数学字体命令，如 \mathbb, \mathcal, \mathfrak，
    # 需要加载 amsfonts（amsmath/amssymb 已经提供部分支持，但 amsfonts 提供更完整的数学字体）
    if re.search(r'\\(mathbb|mathcal|mathfrak)', table_content):
        required_packages.add('amsfonts')
    # 检查矩阵和数组环境，如 \begin{pmatrix} 或 \begin{bmatrix}（需要 amsmath）
    if re.search(r'\\begin\{(pmatrix|bmatrix|matrix|vmatrix|Vmatrix)\}', table_content):
        required_packages.add('amsmath')
    # 检查行内或行间数学公式：使用 $...$ 或 \( ... \)
    if re.search(r'\$.*?\$', table_content) or re.search(r'\\\(', table_content):
        required_packages.add('amsmath')
    # 检查使用 \norm 命令（通常 mathtools 对其进行了扩展）
    if re.search(r'\\norm', table_content):
        required_packages.add('mathtools')
    # 检查高级数学符号，如 \cancel（用于删除线效果），需要 cancel 包
    if re.search(r'\\cancel', table_content):
        required_packages.add('cancel')
    # 检查扩展积分符号，如 \oiint 等，使用 esint 包
    if re.search(r'\\oiint', table_content):
        required_packages.add('esint')
    # 检查 dsfont 命令（如 \mathds，用于黑板粗体符号），需要 dsfont 包
    if re.search(r'\\mathds', table_content):
        required_packages.add('dsfont')
    # 检查双中括号命令，如 \llbracket 和 \rrbracket，加载 stmaryrd 包
    if re.search(r'\\llbracket|\\rrbracket', table_content):
        required_packages.add('stmaryrd')
    # 检查直立希腊字母，如 \upalpha, \upbeta, \upgamma 等，加载 upgreek 包
    if re.search(r'\\up(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)', table_content):
        required_packages.add('upgreek')
    # 检查一些物理学相关的微分运算命令，如 \dv 或 \pdv，使用 physics 包
    if re.search(r'\\(dv|pdv)\b', table_content):
        required_packages.add('physics')

    
    # --- 文本格式化命令 ---
    #if (re.search(r'\\textit', table_content) or
    #    re.search(r'\\textbf', table_content) or
    #    re.search(r'\\underline', table_content) or
    #    re.search(r'\\emph', table_content)):
    #    required_packages.update(['amsmath', 'amssymb'])
    # 如果出现字体颜色调整，也保证 xcolor 加载
    if re.search(r'\\textcolor', table_content):
        required_packages.add('xcolor')
    
    # --- 图形与插图 ---
    # 插入图形命令：\includegraphics, \resizebox, \scalebox
    if re.search(r'\\includegraphics', table_content) or re.search(r'\\resizebox', table_content) or re.search(r'\\scalebox', table_content):
        required_packages.add('graphicx')
    # 图形旋转命令：\rotatebox, \reflectbox
    if re.search(r'\\rotatebox', table_content) or re.search(r'\\reflectbox', table_content):
        required_packages.add('graphicx')
    if re.search(r'\\includegraphics', table_content):
        required_packages.add('graphicx')
    if re.search(r'\\draw\b', table_content) or re.search(r'\\path\b', table_content):
        required_packages.add('tikz')
    # 如果涉及复杂绘图
    if re.search(r'\\begin\{axis\}', table_content):
        required_packages.add('pgfplots')
    # 代码高亮（如果存在 lstlisting 环境）
    if re.search(r'\\begin\{lstlisting\}', table_content):
        required_packages.add('listings')
    # 如果使用 minted 进行代码高亮
    if re.search(r'\\begin\{minted\}', table_content):
        required_packages.add('minted')
    
    # --- 子表格与子标题 ---
    # 检查现代子标题命令，如 \subcaption, \captionof，
    # 以及可能使用的旧版子图/子表格命令，如 \subfigure, \subtable, \subfloat
    if (re.search(r'\\subcaption', table_content) or 
        re.search(r'\\subfigure', table_content) or
        re.search(r'\\subtable', table_content) or
        re.search(r'\\subfloat', table_content)):
        required_packages.add('subcaption')

    
    # --- 符号支持 ---
    # 检查 \mathbb, \mathcal, \mathfrak 等命令，加载 amsfonts
    if re.search(r'\\mathbb', table_content) or re.search(r'\\mathcal', table_content) or re.search(r'\\mathfrak', table_content):
        required_packages.add('amsfonts')
    # 检查希腊字母等常用符号，加载 amssymb
    #if re.search(r'\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\lambda|\\mu', table_content):
    #    required_packages.add('amssymb')
    # 如果使用了 \mathds 命令（常用于黑板粗体），加载 dsfont 包
    if re.search(r'\\mathds', table_content):
        required_packages.add('dsfont')
    # 如果使用了双中括号命令，如 \llbracket 和 \rrbracket，加载 stmaryrd 包
    if re.search(r'\\llbracket|\\rrbracket', table_content):
        required_packages.add('stmaryrd')
    # 如果使用直立希腊字母（例如 \upalpha, \upbeta, \upgamma 等），加载 upgreek 包
    if re.search(r'\\upalpha|\\upbeta|\\upgamma', table_content):
        required_packages.add('upgreek')
    # 勾、叉、圈符号
    if re.search(r'\\checkmark', table_content):
        required_packages.add('amssymb')  # 勾符号
    # ding 符号属于 pifont
    if re.search(r'\\ding\{51\}', table_content):
        required_packages.add('pifont')
    #if re.search(r'\\times', table_content) or re.search(r'\\xmark', table_content):
    #    required_packages.add('amssymb')  # 叉符号
    #if re.search(r'\\circ', table_content):
    #    required_packages.add('amssymb')  # 圈符号
    # xmark 属于 pifont（或自定义）
    if re.search(r'\\xmark', table_content):
        required_packages.add('pifont')
    if re.search(r'\\boxtimes', table_content) or re.search(r'\\square', table_content):
        required_packages.add('amssymb')  # 矩形叉符号
    # 其他包
    if re.search(r'\\ding', table_content):
        required_packages.add('pifont')  # 圆形勾和叉
    # 使用 xmark 或其他圈叉符号的情况
    if re.search(r'\\xmark', table_content) or re.search(r'\\ding\{51\}', table_content):
        required_packages.add('marvosym')  # 圆形叉勾符号

    
    # --- 表格格式与行间距 ---
    # 检查是否调整行间距或单元格高度：\arraystretch, \extrarowheight
    if re.search(r'\\renewcommand\{\\arraystretch\}', table_content):
        required_packages.add('array')
    if re.search(r'\\setlength\{\\extrarowheight\}', table_content):
        required_packages.add('array')
    # 检查是否使用 makecell 相关命令，如 \makecell 或 \thead（常用于单元格内多行文本）
    if re.search(r'\\makecell', table_content) or re.search(r'\\thead', table_content):
        required_packages.add('makecell')
    # 检查自动换行与动态列宽环境：\begin{tabularx}
    if re.search(r'\\begin\{tabularx\}', table_content):
        required_packages.add('tabularx')
    # 绘制线条，适用于数学矩阵和表格混排，功能丰富。
    if re.search(r'\\Nicematrix' in table_content or r'\\begin{NiceTabular}', table_content):
        required_packages.add('nicematrix')
    # 行高行距控制
    #if re.search(r'\\fontsize|\\linespread|\\renewcommand\\arraystretch', table_content):
    if re.search(r'\\doublespacing|\\onehalfspacing|\\singlespacing', table_content):
        required_packages.add('setspace')
    # 检查长表格环境：\begin{longtable}
    if re.search(r'\\begin\{longtable\}', table_content):
        required_packages.add('longtable')

    # ---------- 复杂对齐与排版支持 ----------
    if re.search(r'>{\\raggedright}|>{\\centering}|>{\\RaggedRight}|\\RaggedRight|\\RaggedLeft|\\Centering', table_content):
        required_packages.add('ragged2e')
    if re.search(r'D{.*?}{.*?}{.*?}', table_content):
        required_packages.add('dcolumn')
    if re.search(r'\\SI\{.*?\}\{.*?\}|S{', table_content):  # S 列 or \SI 宏
        required_packages.add('siunitx')
    if re.search(r'\\begin{varwidth}|>{\\begin{varwidth}', table_content):
        required_packages.add('varwidth')


    # --- 线型及元素样式 ---
    # 检查虚线命令：如 \hdashline 或 \cdashline，加载 arydshln 包
    if re.search(r'\\hdashline', table_content) or re.search(r'\\cdashline', table_content):
        required_packages.add('arydshln')
    # 对角线命令支持：diagbox
    if re.search(r'\\diagbox\{.*?\}', table_content):
        required_packages.add('diagbox')
    # 虚线支持：arydshln
    if re.search(r'\\hdashline', table_content) or re.search(r'\\cdashline', table_content):
        required_packages.add('arydshln')
    # 高级横竖线控制：hhline
    if re.search(r'\\hhline\{.*?\}', table_content):
        required_packages.add('hhline')
    # 更粗横线等支持：makecell 的 \Xhline
    if re.search(r'\\Xhline', table_content):
        required_packages.add('makecell')
    # 如果使用 colortbl 进行带颜色的表格边框设置，也可以加载 colortbl
    if re.search(r'\\arrayrulecolor', table_content):
        required_packages.add('colortbl')
    # TikZ 自定义线条（极少用于表格）
    if r'\begin{tikzpicture}' in table_content in table_content:
        required_packages.add('tikz')
    
    # 三线表结构：threeparttable
    if re.search(r'\\begin\{threeparttable\}', table_content) or re.search(r'\\begin\{tablenotes\}', table_content):
        required_packages.add('threeparttable')

    # 自动列宽：tabularx
    if re.search(r'\\begin\{tabularx\}', table_content):
        required_packages.add('tabularx')

    # 智能列宽：tabulary
    if re.search(r'\\begin\{tabulary\}', table_content):
        required_packages.add('tabulary')

    # 多页表格：longtable
    if re.search(r'\\begin\{longtable\}', table_content):
        required_packages.add('longtable')

    # 高级多页表格：ltablex
    if re.search(r'\\begin\{ltablex\}', table_content):
        required_packages.add('ltablex')

    # 单元格换行：makecell
    if re.search(r'\\makecell', table_content) or re.search(r'\\thead', table_content):
        required_packages.add('makecell')

    # 表格旋转：rotating
    if re.search(r'\\begin\{sidewaystable\}', table_content) or re.search(r'\\rotatebox', table_content):
        required_packages.add('rotating')

    # 缩放/调整表格大小：adjustbox
    if re.search(r'\\begin\{adjustbox\}', table_content):
        required_packages.add('adjustbox')

    # 标题控制：caption
    if re.search(r'\\captionof', table_content) or re.search(r'\\captionsetup', table_content):
        required_packages.add('caption')


    
    # --- 页眉与页脚 ---
    if re.search(r'\\fancyhead', table_content) or re.search(r'\\pagestyle\{fancy\}', table_content):
        required_packages.add('fancyhdr')
    
    # --- 其他扩展功能 ---
    # 编码支持（一般在导言区定义，但如果表格代码中出现也做检测）
    if re.search(r'\\usepackage\{inputenc\}', table_content):
        required_packages.add('inputenc')
    # 字体支持
    if re.search(r'\\usepackage\{fontenc\}', table_content):
        required_packages.add('fontenc')
    if re.search(r'\\usepackage\{mathptmx\}', table_content):
        required_packages.add('mathptmx')
    # 浮动体控制 [H]
    if re.search(r'\[H\]', table_content):
        required_packages.add('float')
    
    # --- 高级扩展 ---
    # xparse、etoolbox 等如果有自定义命令可能会用到，但通常不会在表格中直接出现
    
    return required_packages


# RGB 映射表（部分示例，可自行扩展）
html_color_rgb = {
    'aliceblue': (0.94, 0.97, 1.0), 'antiquewhite': (0.98, 0.92, 0.84),
    'aqua': (0.0, 1.0, 1.0), 'aquamarine': (0.44, 1.0, 0.69), 'azure': (0.94, 1.0, 1.0),
    'beige': (0.96, 0.96, 0.86), 'bisque': (1.0, 0.89, 0.77), 'black': (0.0, 0.0, 0.0),
    'blanchedalmond': (1.0, 0.92, 0.8), 'blue': (0.0, 0.0, 1.0), 'blueviolet': (0.54, 0.17, 0.89),
    'brown': (0.65, 0.16, 0.16), 'burlywood': (0.87, 0.72, 0.53), 'cadetblue': (0.37, 0.62, 0.63),
    'chartreuse': (0.5, 1.0, 0.0), 'chocolate': (0.82, 0.41, 0.12), 'coral': (1.0, 0.5, 0.31),
    'cornflowerblue': (0.39, 0.58, 0.93), 'crimson': (0.86, 0.08, 0.24), 'cyan': (0.0, 1.0, 1.0),
    'darkblue': (0.0, 0.0, 0.55), 'darkcyan': (0.0, 0.55, 0.55), 'darkgoldenrod': (0.72, 0.53, 0.04),
    'darkgray': (0.66, 0.66, 0.66), 'darkgreen': (0.0, 0.39, 0.0), 'darkkhaki': (0.74, 0.72, 0.42),
    'darkmagenta': (0.55, 0.0, 0.55), 'darkolivegreen': (0.33, 0.42, 0.18), 'darkorange': (1.0, 0.55, 0.0),
    'darkorchid': (0.6, 0.2, 0.8), 'darkred': (0.55, 0.0, 0.0), 'darksalmon': (0.91, 0.59, 0.48),
    'darkseagreen': (0.56, 0.74, 0.56), 'darkslateblue': (0.28, 0.24, 0.55), 'darkslategray': (0.18, 0.31, 0.31),
    'darkturquoise': (0.0, 0.81, 0.82), 'darkviolet': (0.58, 0.0, 0.83), 'deeppink': (1.0, 0.08, 0.58),
    'deepskyblue': (0.0, 0.75, 1.0), 'dimgray': (0.41, 0.41, 0.41), 'dodgerblue': (0.12, 0.56, 1.0),
    'firebrick': (0.7, 0.13, 0.13), 'floralwhite': (1.0, 0.98, 0.94), 'forestgreen': (0.13, 0.55, 0.13),
    'fuchsia': (1.0, 0.0, 1.0), 'gainsboro': (0.86, 0.86, 0.86), 'ghostwhite': (0.97, 0.97, 1.0),
    'gold': (1.0, 0.84, 0.0), 'goldenrod': (0.85, 0.65, 0.13), 'gray': (0.5, 0.5, 0.5), 'grey': (0.5, 0.5, 0.5),
    'green': (0.0, 1.0, 0.0), 'greenyellow': (0.68, 1.0, 0.18), 'honeydew': (0.94, 1.0, 0.94),
    'hotpink': (1.0, 0.41, 0.71), 'indianred': (0.80, 0.36, 0.36), 'indigo': (0.29, 0.0, 0.51),
    'ivory': (1.0, 1.0, 0.94), 'khaki': (0.94, 0.90, 0.55), 'lavender': (0.90, 0.90, 0.98),
    'lavenderblush': (1.0, 0.94, 0.96), 'lawngreen': (0.49, 0.99, 0.0), 'lemonchiffon': (1.0, 0.98, 0.8),
    'lightblue': (0.68, 0.85, 0.9), 'lightcoral': (0.94, 0.5, 0.5), 'lightcyan': (0.88, 1.0, 1.0),
    'lightgoldenrodyellow': (0.98, 0.98, 0.82), 'lightgreen': (0.56, 0.93, 0.56), 'lightgrey': (0.83, 0.83, 0.83),
    'lightpink': (1.0, 0.71, 0.76), 'lightsalmon': (1.0, 0.63, 0.48), 'lightseagreen': (0.13, 0.7, 0.67),
    'lightskyblue': (0.53, 0.81, 0.98), 'lightslategray': (0.47, 0.53, 0.60), 'lightsteelblue': (0.69, 0.77, 0.87),
    'lightyellow': (1.0, 1.0, 0.88), 'lime': (0.0, 1.0, 0.0), 'limegreen': (0.2, 0.8, 0.2),
    'linen': (0.98, 0.94, 0.90), 'magenta': (1.0, 0.0, 1.0), 'maroon': (0.69, 0.19, 0.38),
    'mediumaquamarine': (0.4, 0.8, 0.67), 'mediumblue': (0.0, 0.0, 0.8), 'mediumorchid': (0.73, 0.33, 0.83),
    'mediumpurple': (0.58, 0.44, 0.86), 'mediumseagreen': (0.24, 0.7, 0.44), 'mediumslateblue': (0.48, 0.41, 0.94),
    'mediumspringgreen': (0.0, 0.98, 0.6), 'mediumturquoise': (0.28, 0.82, 0.8), 'mediumvioletred': (0.78, 0.08, 0.52),
    'midnightblue': (0.1, 0.1, 0.44), 'mintcream': (0.96, 1.0, 0.98), 'mistyrose': (1.0, 0.89, 0.88),
    'moccasin': (1.0, 0.89, 0.71), 'navajowhite': (1.0, 0.87, 0.68), 'navy': (0.0, 0.0, 0.5),
    'oldlace': (0.99, 0.96, 0.9), 'olive': (0.5, 0.5, 0.0), 'olivedrab': (0.42, 0.56, 0.14),
    'orange': (1.0, 0.65, 0.0), 'orangered': (1.0, 0.27, 0.0), 'orchid': (0.85, 0.44, 0.84),
    'palegoldenrod': (0.93, 0.91, 0.67), 'palegreen': (0.6, 0.98, 0.6), 'paleturquoise': (0.68, 0.93, 0.93),
    'palevioletred': (0.86, 0.44, 0.58), 'papayawhip': (1.0, 0.94, 0.84), 'peachpuff': (1.0, 0.85, 0.73),
    'peru': (0.8, 0.52, 0.25), 'pink': (1.0, 0.75, 0.8), 'plum': (0.87, 0.63, 0.87),
    'powderblue': (0.69, 0.88, 0.9), 'purple': (0.63, 0.13, 0.94), 'rebeccapurple': (0.4, 0.2, 0.6),
    'red': (1.0, 0.0, 0.0), 'rosybrown': (0.74, 0.56, 0.56), 'royalblue': (0.25, 0.41, 0.88),
    'saddlebrown': (0.55, 0.27, 0.07), 'salmon': (0.98, 0.5, 0.45), 'sandybrown': (0.96, 0.64, 0.38),
    'seashell': (1.0, 0.96, 0.93), 'sienna': (0.63, 0.32, 0.18), 'silver': (0.75, 0.75, 0.75),
    'skyblue': (0.53, 0.81, 0.92), 'slateblue': (0.42, 0.35, 0.8), 'slategray': (0.44, 0.5, 0.56),
    'snow': (1.0, 0.98, 0.98), 'springgreen': (0.0, 1.0, 0.5), 'steelblue': (0.27, 0.51, 0.71),
    'tan': (0.82, 0.71, 0.55), 'teal': (0.0, 0.5, 0.5), 'thistle': (0.85, 0.75, 0.85),
    'tomato': (1.0, 0.39, 0.28), 'turquoise': (0.25, 0.87, 0.82), 'violet': (0.93, 0.51, 0.93),
    'wheat': (0.96, 0.87, 0.7), 'white': (1.0, 1.0, 1.0), 'whitesmoke': (0.96, 0.96, 0.96),
    'yellow': (1.0, 1.0, 0.0), 'yellowgreen': (0.6, 0.8, 0.2)
}

def wrap_table_with_latex(table_content):
    """
    封装为 LaTeX 文档，自动引入 T1 编码和颜色定义（使用 RGB 定义）。
    """

    # 提取颜色
    color_pattern = re.compile(r'\\(?:cellcolor|textcolor|rowcolor)(?:\[.*?\])?{(\w+)}')
    used_colors = set(color_pattern.findall(table_content))

    # xcolor 默认色
    xcolor_builtin = {
        'white', 'black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
        'gray', 'darkgray', 'lightgray', 'brown', 'lime', 'olive', 'orange',
        'pink', 'purple', 'teal', 'violet'
    }
    custom_colors = used_colors - xcolor_builtin


    # 构造颜色定义（RGB）
    define_colors_str = ""
    for color in sorted(custom_colors):
        if color.lower() in html_color_rgb:
            r, g, b = html_color_rgb[color.lower()]
            define_colors_str += f"\\definecolor{{{color}}}{{rgb}}{{{r:.2f}, {g:.2f}, {b:.2f}}}\n"
        else:
            define_colors_str += f"% [警告] 未识别颜色: {color}\n"

    packages_str = "\\usepackage{adjustbox}\n\\usepackage{color}\n\\usepackage{graphics}\n\\usepackage{pifont}\n\\usepackage{amssymb,amsmath}\n\\usepackage{multirow,multicol}\n\\usepackage{booktabs}\n\\usepackage{tabularx}\n\\usepackage{subfig}\n\\usepackage{caption}\n\\usepackage{siunitx}\n\\usepackage{array}\n\\usepackage{bm}\n\\usepackage[table]{xcolor}\n\\usepackage{makecell}\n\\definecolor{lightblue}{rgb}{0.68, 0.85, 0.90}\n\\definecolor{darkgreen}{rgb}{0.00, 0.39, 0.00}\n\\definecolor{ForestGreen}{rgb}{0.13, 0.55, 0.13}\n\\definecolor{Gray}{rgb}{0.50, 0.50, 0.50}"

    table_content = re.sub(
        r'(\\begin\{tabular\})',
        r'\\begin{adjustbox}{max width=\\textwidth, max height=0.8\\textheight, keepaspectratio}\n\1',
        table_content,
        count=1
    )
    # 找到所有 \end{tabular} 的位置
    matches = list(re.finditer(r'(\\end\{tabular\})', table_content))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span()
        # 在最后一个 \end{tabular} 后插入 \end{adjustbox}
        table_content = (
            table_content[:end] + '\n\\end{adjustbox}' + table_content[end:]
        )
    # 最终模板
    template = f"""\\documentclass{{article}}
{packages_str}
\\pagestyle{{empty}}
\\begin{{document}}
{table_content}
\\end{{document}}
"""
    return template


def wrap_table_with_adjustbox(table_content):
    # 确保添加 adjustbox 宏包和 \pagestyle{empty}
    if "\\begin{document}" in table_content:
        # 在 \begin{document} 前添加 usepackage 和 pagestyle（若未添加过）
        if "\\usepackage{adjustbox}" not in table_content:
            table_content = re.sub(
                r"(\\begin\{document\})",
                r"\\usepackage{adjustbox}\n\\pagestyle{empty}\n\1",
                table_content,
                count=1
            )
        elif "\\pagestyle{empty}" not in table_content:
            table_content = re.sub(
                r"(\\begin\{document\})",
                r"\\pagestyle{empty}\n\1",
                table_content,
                count=1
            )

    # 包裹第一个 tabular 环境
    table_content = re.sub(
        r'(\\begin\{tabular\})',
        r'\\begin{adjustbox}{max width=\\textwidth, max height=0.8\\textheight, keepaspectratio}\n\1',
        table_content,
        count=1
    )

    # 找到所有 \end{tabular} 的位置
    matches = list(re.finditer(r'(\\end\{tabular\})', table_content))
    if matches:
        last_match = matches[-1]
        start, end = last_match.span()
        # 在最后一个 \end{tabular} 后插入 \end{adjustbox}
        table_content = (
            table_content[:end] + '\n\\end{adjustbox}' + table_content[end:]
        )

    return table_content


def extract_tabular_format(latex_code: str) -> str:
    pattern = r'\\begin\{tabular\}'
    match = re.search(pattern, latex_code)
    if not match:
        return None

    start_idx = match.end()
    # 从 start_idx 开始找到第一个完整的大括号内容（支持嵌套）
    i = start_idx
    while i < len(latex_code) and latex_code[i] != '{':
        i += 1
    if i == len(latex_code):
        return None
    
    brace_count = 0
    fmt = ''
    i += 1  # skip first {
    while i < len(latex_code):
        char = latex_code[i]
        if char == '{':
            brace_count += 1
            fmt += char
        elif char == '}':
            if brace_count == 0:
                break
            brace_count -= 1
            fmt += char
        else:
            fmt += char
        i += 1
    return fmt

def is_valid_format(fmt: str) -> bool:
    # 只允许 l, r, c, @, {}, 空格
    allowed_chars = set('|lrc@{} ')
    return all(char in allowed_chars for char in fmt)

def remove_at_braces(fmt: str) -> str:
    # 删除所有的 @{...} 结构
    return re.sub(r'@{[^{}]*}', '', fmt)

def clean_between(latex_code: str, begin: str, end: str) -> str:
    pattern = rf'({re.escape(begin)})(.*?)(?={re.escape(end)})'
    match = re.search(pattern, latex_code, re.DOTALL)
    if not match:
        return latex_code

    before = match.group(1)
    middle = match.group(2)

    cleaned_lines = []
    for line in middle.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue  # 删除空行
        #if stripped.startswith(r'\centering') or re.match(r'\\caption\{.*\}', stripped):
        if re.match(r'\\caption\{.*\}', stripped):
            cleaned_lines.append(stripped)
    
    #has_centering = any(r'\begin{center}' in line.strip() for line in middle.strip().splitlines())

    #if has_centering:
    #    cleaned_lines.insert(0, r'\centering')
    if cleaned_lines:
        cleaned_middle = '\n'.join(cleaned_lines)
        replacement = f'{before}\n{cleaned_middle}\n'
    else:
        replacement = f'{before}\n'  # 只保留前缀，不插入空行

    latex_code = latex_code.replace(match.group(0), replacement)
    return latex_code

def extract_brace_content(text, start_idx):
    """ 提取 text[start_idx] 开始的 {...}，支持嵌套，返回 (内容, 下一个位置) """
    if text[start_idx] != '{':
        return None, start_idx
    i = start_idx + 1
    brace_count = 1
    content = ''
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return content, i + 1
        content += text[i]
        i += 1
    return None, start_idx  # 括号不匹配

def replace_multicolumn_1(latex_code):
    # 匹配 \multicolumn {1} {对齐} { 的开头部分（宽容空格）
    pattern = re.compile(r'\\multicolumn\s*\{\s*1\s*\}\s*\{\s*[^{}]*\s*\}\s*\{', re.DOTALL)
    
    result = ''
    last_idx = 0

    for match in pattern.finditer(latex_code):
        start = match.start()
        brace3_start = match.end() - 1  # 第三个 `{` 的位置

        # 提取第三个 {...}
        content, next_idx = extract_brace_content(latex_code, brace3_start)
        if content is None:
            continue

        # 替换整段为 content
        result += latex_code[last_idx:start] + content
        last_idx = next_idx

    result += latex_code[last_idx:]
    return result

def clean_latex_spacing(latex_code: str) -> str:
    # 删除 \vspace{-4mm} 和 \vspace*{-4mm}
    latex_code = re.sub(r'\\vspace\*?\s*{[^}]*}', '', latex_code)

    # 删除 \addlinespace[3pt]
    latex_code = re.sub(r'\\addlinespace\s*\[[^\]]*\]', '', latex_code)

    # 删除 \\[4pt] 形式，替换为 \\
    latex_code = re.sub(r'\\\\\s*\[[^\]]*\]', r'\\\\', latex_code)

    # 删除 \rule{0pt}{4ex}
    latex_code = re.sub(r'\\rule\s*{[^}]*}\s*{[^}]*}', '', latex_code)

    return latex_code

def normalize_whitespace(latex_code: str) -> str:
    """
    清理 LaTeX 代码中的空白：
    1. 两个以上的空格替换成一个空格；
    2. 两个以上的换行符替换成一个换行符；
    """
    # 1. 替换多个空格为一个空格（注意不替换制表符或换行）
    latex_code = re.sub(r'[ ]{2,}', ' ', latex_code)

    # 2. 替换多个换行为一个换行
    latex_code = re.sub(r'\n{2,}', '\n', latex_code)

    return latex_code

def get_valid_table(latex_code: str) -> str:
    # 如果包含其他 tabular 环境类型，立即返回 None
    disallowed_envs = ['tabularx', 'tabular*', 'tabulary', 'longtable', 'array', 'tabu']
    for env in disallowed_envs:
        if re.search(r'\\begin\{' + re.escape(env) + r'\}', latex_code):
            return None
        
    # \begin{tabular} 数量检查
    tabular_count = len(re.findall(r'\\begin\{tabular\}', latex_code))
    if tabular_count > 1:
        return None

    # 检查是否包含 \makecell
    if r'\makecell' in latex_code:
        return None\
    
    # ✅ 删除 tabular [c] 的空格，如 \begin{tabular} [c] → \begin{tabular}[c]
    latex_code = re.sub(r'(\\begin\{tabular\})\s+\[', r'\1[', latex_code)
    # ✅ 删除 tabular {...} 的空格，如 \begin{tabular} {c|c} → \begin{tabular}{c|c}
    latex_code = re.sub(r'(\\begin\{tabular\}(?:\[[^\]]*\])?)\s+\{', r'\1{', latex_code)
    
    latex_code = re.sub(r'(\\begin\{tabular\})\[[^\]]*\]', r'\1', latex_code)

    # 提取格式串
    fmt = extract_tabular_format(latex_code)

    if fmt is None or not is_valid_format(fmt):
        return None

    # 删除 @{} 后的格式串
    new_fmt = remove_at_braces(fmt)
    new_fmt = new_fmt.replace(' ', '')
    
    # 替换原格式串（确保精确匹配 \begin{tabular}{fmt}）
    pattern = r'(\\begin\{tabular\})\{' + re.escape(fmt) + r'\}'
    latex_code = re.sub(pattern, r'\1{' + new_fmt + r'}', latex_code)

    # 清理 \begin{table} 和 \begin{tabular} 之间
    latex_code = clean_between(latex_code, r'\begin{table}', r'\begin{tabular}')

    # 清理 \end{tabular} 和 \end{table} 之间
    latex_code = clean_between(latex_code, r'\end{tabular}', r'\end{table}')
    latex_code = re.sub(
        r'(\\begin\{table\})',
        r'\1\n\\centering',
        latex_code,
        count=1
    )
    latex_code = replace_multicolumn_1(latex_code)
    latex_code = clean_latex_spacing(latex_code)
    latex_code = normalize_whitespace(latex_code)
    return latex_code.strip()

pred = r"""\begin{table}
\begin{tabular}{cc}    
Name      & Age    \\


Alice     &   23    \\


Bob       &  31    \\
\end{tabular}
\centering
\end{table}
"""
print(get_valid_table(pred))


def process_tex_files(root_dir, output_base_dir):
    """
    遍历所有 tex 文件，提取表格并保存为可编译的 LaTeX 文件
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    total_tex_files = 0  # 总的tex文件数
    successful_tex_files = 0  # 成功处理的tex文件数
    failed_tex_files = 0  # 处理失败的tex文件数
    
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        output_folder = os.path.join(output_base_dir, f"{folder_name}_table")
        os.makedirs(output_folder, exist_ok=True)
        
        table_count = 0

        # 第一步：遍历文件夹中的所有 tex 文件，提取表格并保存为新的 LaTeX 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".tex"):
                total_tex_files += 1
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # 提取该文件中的表格
                    tables = extract_tables_from_tex(content)
                    for table in tables:
                        table = get_valid_table(table)
                        if table is None:
                            continue
                        table_count += 1
                        # 根据表格内容动态确定所需的包
                        #required_packages = analyze_table_for_required_packages(table)

                        # 封装表格并加入相关包
                        wrapped_table = wrap_table_with_latex(table)

                        # 保存新的表格文件
                        output_file = os.path.join(output_folder, f"{folder_name}_table{table_count}.tex")
                        with open(output_file, 'w', encoding='utf-8') as out_file:
                            out_file.write(wrapped_table)
                    
                    # 成功处理的tex文件数
                    successful_tex_files += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    # 处理失败的tex文件数
                    failed_tex_files += 1
    
    # 输出成功处理和失败的tex文件数
    print(f"Total tex files: {total_tex_files}")
    print(f"Successfully processed tex files: {successful_tex_files}")
    print(f"Failed tex files: {failed_tex_files}")

    print("表格提取并转换完成！")

def get_table_tex(input_tex, output_file):
    try:
        # 提取该文件中的表格
        #table = extract_tables_from_tex(input_tex)[0]

        # 封装表格并加入相关包
        wrapped_table = wrap_table_with_adjustbox(input_tex)

        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(wrapped_table)
        
        return True
        
    except Exception as e:
        return False
