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
        self.latex_code = remove_table_environments(latex_code)
        self.root = Node("table")
        self.include_misc = include_misc
        self.include_caption = include_caption

    def parse(self):
        parts = self._split_by_tabular_blocks(self.latex_code)
        for kind, content in parts:
            if kind.startswith("tabular"):
                tabular_match = re.match(
                    r"\\begin\{(tabularx?|tabu|array)\}(\{[^}]*\})?\{([^}]*)\}(.*?)\\end\{\1\}",
                    content, re.DOTALL
                )
                if tabular_match:
                    _, _, fmt, body = tabular_match.groups()
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
            rf"\\begin\{{({'|'.join(tabular_types)})\}}(?:\{{[^}}]*\}})?\{{[^}}]*\}}.*?\\end\{{\1\}}",
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
                    Node("caption", parent=self.root, content=caption_match.group(1))
            elif self.include_misc:
                Node("misc", parent=self.root, content=line)

    def _add_format_node(self, format_string: str, parent_node: Node):
        fmt_node = Node("format", parent=parent_node)
        # 忽略空格，只保留有效格式字符或结构
        pattern = r"(>{[^}]*})|(p{[^}]*})|[lcrXS|!<>@]"
        for match in re.finditer(pattern, format_string):
            Node("fmt_item", parent=fmt_node, content=match.group())

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
                cell_node = Node("cell", parent=row_node, content=cell_code)
                for nested_tabular in self._extract_all_nested_tabular(cell_code):
                    sub_tree = LatexTableParser(nested_tabular, include_misc=self.include_misc, include_caption=self.include_caption).parse()
                    sub_tree.name = "subtable"
                    sub_tree.parent = cell_node

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
    '''
    print("Tree 1 Structure:")
    print(latex1)
    for pre, _, node in RenderTree(tree1):
        print(f"{pre}{node}")

    print("\nTree 2 Structure:")
    print(latex2)
    for pre, _, node in RenderTree(tree2):
        print(f"{pre}{node}")
    '''
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
    return match.group(0)

def compute_teds_score(pred_table, label):
    pred_table = extract_latex_table(pred_table)
    edit_distance, reward = compute_table_similarity(pred_table, label)
    return reward

