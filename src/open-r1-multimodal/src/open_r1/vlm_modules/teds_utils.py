import re
from anytree import Node, RenderTree
from apted import APTED, Config
from apted.helpers import Tree

def remove_table_environments(latex_code: str) -> str:
    # 使用正则表达式找到所有 table 环境
    ENVIRONMENTS = [
        'table', 'table*', 'longtable', 'threeparttable', 'wraptable',
        'sidewaystable', 'subtable', 'supertabular', 'xtabular'
    ]
    blocks = []
    for env in ENVIRONMENTS:
        env_escaped = re.escape(env)
        pattern = rf"\\begin\{{{env_escaped}\}}(?:\[[^\]]*\])?.*?\\end\{{{env_escaped}\}}"
        blocks.extend(re.findall(pattern, latex_code, flags=re.DOTALL))
    return "\n\n".join(blocks)
class LatexTableParser:
    def __init__(self, latex_code: str, include_misc: bool = True, include_caption: bool = True):
        self.latex_code = self.replace_multicolumn_1(remove_table_environments(latex_code))
        self.root = Node("table")
        self.include_misc = include_misc
        self.include_caption = include_caption
    
    def extract_brace_content(self,s: str, start: int):
        """
        提取从 s[start] 开始的第一个 {...}（支持嵌套），返回 (内容, 下一个位置)
        """
        if s[start] != '{':
            return None, start
        i = start + 1
        depth = 1
        content = []
        while i < len(s):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return ''.join(content), i + 1
            content.append(s[i])
            i += 1
        return None, start
    
    def replace_multicolumn_1(self, latex_code):
        # 匹配 \multicolumn {1} {对齐} { 的开头部分（宽容空格）
        pattern = re.compile(r'\\multicolumn\s*\{\s*1\s*\}\s*\{\s*[^{}]*\s*\}\s*\{', re.DOTALL)
        
        result = ''
        last_idx = 0

        for match in pattern.finditer(latex_code):
            start = match.start()
            brace3_start = match.end() - 1  # 第三个 `{` 的位置

            # 提取第三个 {...}
            content, next_idx = self.extract_brace_content(latex_code, brace3_start)
            if content is None:
                continue

            # 替换整段为 content
            result += latex_code[last_idx:start] + content
            last_idx = next_idx

        result += latex_code[last_idx:]
        return result

    
    def extract_tabular_env(self, content: str):
        """
        提取 tabular/tabularx/tabu/array 环境内容，忽略可选参数 []，提取 format_string 和正文。
        返回：(env_type, format_string, body)
        """
        import re

        # 匹配 \begin{env}[可选]{格式} 的开头
        begin_pattern = r"\\begin\{(tabularx?|tabu|array)\}\s*(?:\[[^\]]*\]\s*)?\{"
        begin_match = re.match(begin_pattern, content)
        if not begin_match:
            return None

        env_type = begin_match.group(1)
        i = begin_match.end() - 1  # 当前位于 format_string 的起始 `{`

        # 提取 format_string
        format_string, i = self.extract_brace_content(content, i)

        # 提取正文，直到 \end{env_type}
        end_pattern = f"\\end{{{env_type}}}"
        end_index = content.find(end_pattern, i)
        if end_index == -1:
            return None

        body = content[i:end_index]

        return env_type, format_string, body.strip()



    def parse(self):
        parts = self._split_by_tabular_blocks(self.latex_code)
        for kind, content in parts:
            if kind.startswith("tabular"):
                '''
                tabular_match = re.match(
                    r"\\begin\{(tabularx?|tabu|array)\}(\{[^}]*\})?\{([^}]*)\}(.*?)\\end\{\1\}",
                    content, re.DOTALL
                )
                if tabular_match:
                '''
                result = self.extract_tabular_env(content)
                if result:
                    env_type, fmt, body = result

                    #_, _, fmt, body = tabular_match.groups()
                    tabular_node = Node("tabular", parent=self.root)  # unified as "tabular"
                    self._add_format_node(fmt, tabular_node)
                    self._parse_single_tabular(fmt, body, tabular_node)
            elif kind == "other":
                self._parse_misc_lines(content)
        return self.root

    def _split_by_tabular_blocks(self, text):
        parts = []
        tabular_types = ["tabular", "tabularx", "tabu", "array"]
        pattern = re.compile(
            rf"\\begin\{{({'|'.join(tabular_types)})\}}\s*(?:\[[^\]]*\]\s*)?\{{[^}}]*\}}.*?\\end\{{\1\}}",
            re.DOTALL
        )
        last_index = 0
        for match in pattern.finditer(text):
            if match.start() > last_index:
                parts.append(("other", text[last_index:match.start()]))
            parts.append(("tabular", match.group(0)))
            last_index = match.end()
        if last_index < len(text):
            parts.append(("other", text[last_index:]))
        return parts

    def _parse_misc_lines(self, content):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            if line.startswith(r"\caption") and self.include_caption:
                caption_match = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", line)
                if caption_match:
                    Node("caption", parent=self.root, content=caption_match.group(1).strip())
            elif self.include_misc:
                Node("misc", parent=self.root, content=line)

    def _add_format_node(self, format_string: str, parent_node: Node):
        fmt_node = Node("format", parent=parent_node)
        # 忽略空格，只保留有效格式字符或结构
        #print(f"format_string{format_string}")
        #pattern = r"(>{[^}]*})|(p{[^}]*})|[lcrXS|!<>@]"
        pattern = r"(>{[^}]*})|(p{[^}]*})|(@{[^}]*})|[lcrXS|!<>@]"
        '''for match in re.finditer(pattern, format_string):
            Node("fmt_item", parent=fmt_node, content=match.group())'''
        if format_string is None:
            print("[Warning] format_string is None, skipping format parsing.")
            return

        if fmt_node is None:
            print("[Warning] fmt_node is None, skipping format node attachment.")
            return

        for match in re.finditer(pattern, format_string):
            try:
                content = match.group()
                if content:
                    Node("fmt_item", parent=fmt_node, content=content)
                else:
                    print("[Warning] Empty content in format string match.")
            except Exception as e:
                print(f"[Warning] Failed to create fmt_item node: {e}")

    def _parse_single_tabular(self, format_string: str, tabular_content: str, parent_node: Node):
        tabular_content = re.sub(r"\\tabularnewline", r"\\\\", tabular_content)
        tabular_content = re.sub(r"\\\\\s*\n", r"\\\\", tabular_content)
        row_split_pattern = r'(?<!\\)\\(?:\\|\*|\[|\])'
        rows = re.split(row_split_pattern, tabular_content)
        for row in rows:
            line = row.strip()
            if not line:
                row_node = Node("row", parent=parent_node)
                continue
            style_cmds = [
                r"\\addlinespace", r"\\arraystretch\{[^}]+\}",
                r"\\renewcommand\{\\arraystretch\}\{[^}]+\}", r"\\noalign\{[^}]+\}",
                r"\\vspace\*?\{[^}]+\}", r"\\hspace\*?\{[^}]+\}", r"\\specialrule\{[^}]+\}\{[^}]*\}\{[^}]*\}"
            ]
            for pattern in style_cmds:
                match = re.search(pattern, line)
                if match:
                    Node("line", parent=parent_node, content=match.group())
                    line = line.replace(match.group(), "").strip()
            if line.startswith(r"\toprule"):
                Node("line", parent=parent_node, content=r"\toprule")
                line = line[len(r"\toprule"):].strip()
                if not line:
                    continue
            if line.startswith(r"\midrule"):
                Node("line", parent=parent_node, content=r"\midrule")
                line = line[len(r"\midrule"):].strip()
                if not line:
                    continue
            if line.startswith(r"\bottomrule"):
                Node("line", parent=parent_node, content=r"\bottomrule")
                line = line[len(r"\bottomrule"):].strip()
                if not line:
                    continue
            if line.startswith(r"\hline"):
                Node("line", parent=parent_node, content=r"\hline")
                line = line[len(r"\hline"):].strip()
                if not line:
                    continue
            if r"\hline" in line:
                Node("line", parent=parent_node, content=r"\hline")
                line = line.replace(r"\hline", "").strip()
            cline_matches = re.findall(r"(\\cline\{\d+-\d+\})", line)
            for cline in cline_matches:
                Node("line", parent=parent_node, content=cline)
                line = line.replace(cline, "").strip()
            cmidrule_matches = re.findall(r"(\\cmidrule(?:\([lr]*\))?\{\d+-\d+\})", line)
            for cmidrule in cmidrule_matches:
                Node("line", parent=parent_node, content=cmidrule)
                line = line.replace(cmidrule, "").strip()
            hhline_match = re.search(r"(\\hhline\{[^}]*\})", line)
            if hhline_match:
                Node("line", parent=parent_node, content=hhline_match.group(1))
                line = line.replace(hhline_match.group(1), "").strip()
            elif line.startswith(r"\hhline"):
                Node("line", parent=parent_node, content=r"\hhline")
                line = line.replace(r"\hhline", "").strip()
                if not line:
                    continue
            if not line:
                continue
            row_node = Node("row", parent=parent_node)
            cells = self._split_cells(line)
            for cell_code in cells:
                cell_code = self.replace_and_clean_math(cell_code)
                cell_node = Node("cell", parent=row_node, content=cell_code)
                for nested_tabular in self._extract_all_nested_tabular(cell_code):
                    sub_tree = LatexTableParser(nested_tabular, include_misc=self.include_misc, include_caption=self.include_caption).parse()
                    sub_tree.name = "subtable"
                    sub_tree.parent = cell_node

    def replace_and_clean_math(self, content: str) -> str:
        # ① 处理 $...$：替换为 \(...\)，并去除内部空格
        def replace_dollar_math(m):
            inner = m.group(1).replace(" ", "").strip()
            if re.fullmatch(r'-?\d+(\.\d+)?', inner):  # 是纯数字
                return inner
            else:
                return r"\(" + inner + r"\)"

        # 替换 $...$（跳过 $$...$$）
        content = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', replace_dollar_math, content, flags=re.DOTALL)

        # ② 处理 \(...\)：去掉内部空格并处理纯数字
        def clean_paren_math(m):
            inner = m.group(1).replace(" ", "").strip()
            if re.fullmatch(r'-?\d+(\.\d+)?', inner):
                return inner
            else:
                return r"\(" + inner + r"\)"

        # 允许 \(...\) 中间有任意空格，包括末尾的 ` \)`
        content = re.sub(r'\\\(\s*(.+?)\s*\\\)', clean_paren_math, content, flags=re.DOTALL)

        return content



    def _split_cells(self, line: str):
        cells = []
        current = ''
        depth = 0
        nested_tabular_level = 0
        i = 0
        while i < len(line):
            if line[i:i+15] == '\\begin{tabular':
                nested_tabular_level += 1
            if line[i:i+13] == '\\end{tabular}':
                nested_tabular_level = max(0, nested_tabular_level - 1)
            char = line[i]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            if char == '&' and depth == 0 and nested_tabular_level == 0:
                split_cells = self._split_rowcolor(current.strip())
                cells.extend(split_cells)
                current = ''
            else:
                current += char
            i += 1
        current = current.rstrip('\\').strip()
        if current:
            split_cells = self._split_rowcolor(current)
            cells.extend(split_cells)
        return cells

    def _split_rowcolor(self, cell_code: str):
        # 合并 \rowcolor[...]{...} 和后续内容，若内容为空则添加一个空 cell
        match = re.match(r"(\\rowcolor(?:\[[^\]]+\])?\{[^}]+\})(.*)", cell_code, re.DOTALL)
        if match:
            part1 = match.group(1).strip()
            part2 = re.sub(r"[\s\t\n]+", "", match.group(2))
            if not part2:
                return [part1, ""]
            return [part1 + part2]
        else:
            return [cell_code.strip()]

    def _extract_all_nested_tabular(self, text: str):
        pattern = re.compile(r"\\begin\{tabular\}\{[^}]*\}.*?\\end\{tabular\}", flags=re.DOTALL)
        return pattern.findall(text)

class APTEDTreeBuilder:
    @staticmethod
    def anytree_to_apted_tree(node: Node) -> Tree:
        label = node.name
        attributes = {}

        if hasattr(node, 'content'):
            attributes['content'] = node.content

        apted_node = Tree({'label': label, 'attributes': attributes})
        for child in node.children:
            child_apted_node = APTEDTreeBuilder.anytree_to_apted_tree(child)
            apted_node.children.append(child_apted_node)
        return apted_node

class LatexAPTEDConfig(Config):
    def normalized_label(self, node):
        return node.label['label']

    def insert_cost(self, node):
        return 1

    def remove_cost(self, node):
        return 1

    def update_cost(self, a, b):
        return 1  # ✅ 任何替换操作都统一代价 1

def print_tree_structure(tree, indent=0):
    """递归打印树的结构，包括节点和子节点"""
    indent_str = '  ' * indent  # 根据缩进级别创建缩进字符串
    print(f"{indent_str}Node: {tree.label['label']}, Attributes: {tree.label['attributes']}")
    for child in tree.children:
        print_tree_structure(child, indent + 1)

def compute_table_similarity(latex1: str, latex2: str) -> tuple:
    tree1 = LatexTableParser(latex1,include_misc=False, include_caption = True).parse()
    tree2 = LatexTableParser(latex2,include_misc=False, include_caption = True).parse()
    
    #print("Tree 1 Structure:")
    #print(latex1)
    #for pre, _, node in RenderTree(tree1):
    #    print(f"{pre}{node}")

    #print("\nTree 2 Structure:")
    #print(latex2)
    #for pre, _, node in RenderTree(tree2):
    #    print(f"{pre}{node}")
    
    apted_tree1 = APTEDTreeBuilder.anytree_to_apted_tree(tree1)
    apted_tree2 = APTEDTreeBuilder.anytree_to_apted_tree(tree2)

    apted = APTED(apted_tree1, apted_tree2, LatexAPTEDConfig())
    edit_distance = apted.compute_edit_distance()

    num_nodes_1 = sum(1 for _ in RenderTree(apted_tree1))
    num_nodes_2 = sum(1 for _ in RenderTree(apted_tree2))

    reward = 1 - (edit_distance / max(num_nodes_1, num_nodes_2))
    #print(f"edit_distance{edit_distance}")

    return edit_distance, reward


def wrap_table_with_latex(table_content):
    """
    封装为 LaTeX 文档，自动引入 T1 编码和颜色定义（使用 RGB 定义）。
    """
    packages_str = "\\usepackage{color}\n\\usepackage{graphics}\n\\usepackage{pifont}\n\\usepackage{amssymb,amsmath}\n\\usepackage{multirow,multicol}\n\\usepackage{booktabs}\n\\usepackage{tabularx}\n\\usepackage{subfig}\n\\usepackage{caption}\n\\usepackage{siunitx}\n\\usepackage{array}\n\\usepackage{bm}\n\\usepackage[table]{xcolor}\n\\usepackage{makecell}"

    # 最终模板
    template = f"""\\documentclass{{article}}
{packages_str}
\\pagestyle{{empty}}
\\begin{{document}}
{table_content}
\\end{{document}}
"""
    return template

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

# def extract_latex_tabular(latex_code):
#     """
#     提取 LaTeX 字符串中第一个 \begin{table} 与 \end{table} 之间的内容。
#     """
#     pattern = r"\\begin{tabular}.*?\\end{tabular}"
#     match = re.search(pattern, latex_code, re.DOTALL)
#     if match is not None:
#         return match.group(0)
#     else:
#         print("Warning: No LaTeX tabular found in prediction.")
#         return None   

def extract_latex_tabular(latex_code):
    """
    提取 LaTeX 字符串中第一个 \begin{tabular*} 类环境（包括 tabular、tabularx、tabu、array 等）之间的内容。
    """
    # 用正则匹配多种表格环境
    pattern = (
        r"\\begin{(tabular\*?|tabularx|tabu|array|longtable|supertabular)}"
        r".*?"
        r"\\end{\1}"
    )
    match = re.search(pattern, latex_code, re.DOTALL)
    if match is not None:
        return match.group(0)
    else:
        print("Warning: No LaTeX tabular-like environment found in prediction.")
        return None


def compute_teds_score(pred_table, label):
    if pred_table is None or label is None:
        return 0,0
    
    pred_table = extract_latex_table(pred_table)
    label = extract_latex_table(label)

    if pred_table is None or label is None:
        return 0,0

    edit_distance, reward = compute_table_similarity(pred_table, label)
    if reward<0:
        reward = 0
    return edit_distance, reward

def remove_at_braces(s: str) -> str:
    """去除字符串中的 @{ }"""
    if not isinstance(s, str):
        return s
    return s.replace("@{}", "")

def compute_teds_score_only_tabular(pred_table, label):
    if pred_table is None or label is None:
        return 0,0
    
    pred_table = extract_latex_tabular(pred_table)
    label = extract_latex_tabular(label)

    pred_table = remove_at_braces(pred_table)
    label = remove_at_braces(label)

    if pred_table is None or label is None:
        return 0,0
    else:
        pred_table = "\\begin{table}\n" + pred_table + "\n\\end{table}"
        label = "\\begin{table}\n" +  label + "\n\\end{table}"

    edit_distance, reward = compute_table_similarity(pred_table, label)
    if reward<0:
        reward = 0
    return edit_distance, reward

