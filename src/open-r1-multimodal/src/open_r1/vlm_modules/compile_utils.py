import os
import subprocess




def process_tex_file(output_dir='pred_latex', image_filename='default.png'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.splitext(image_filename)[0]
    tex_path = os.path.join(output_dir, f"{base_name}.tex")
    pdflatex_log_path = os.path.join(output_dir, f"{base_name}.pdflatex.log")
    pdf_path = tex_path.replace(".tex", ".pdf")
    pdf_path = pdf_path.replace("\\", "/")
    aux_path = pdf_path.replace(".pdf", ".aux")
    log_path = pdf_path.replace(".pdf", ".log")
    out_path = pdf_path.replace(".pdf", ".out")
    
    try:
        # 1. 运行 pdflatex 转换为 PDF
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path], 
                        cwd=output_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except:
        if os.path.exists(tex_path):
            os.remove(tex_path)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(aux_path):
            os.remove(aux_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        return 0.0
    # try:
    #     # 用 capture_output=True 获取 stdout/stderr，text=True 以字符串形式获取
    #     proc = subprocess.run(
    #         ["pdflatex", "-interaction=nonstopmode", tex_path],
    #         cwd=output_dir,
    #         capture_output=True,
    #         text=True,
    #         timeout=60
    #     )
    #     # 把 pdflatex 的输出写到日志，便于查看原因
    #     with open(pdflatex_log_path, "w", encoding="utf-8") as f:
    #         f.write("=== pdflatex returncode: %s ===\n\n" % proc.returncode)
    #         f.write("=== STDOUT ===\n")
    #         f.write(proc.stdout or "")
    #         f.write("\n\n=== STDERR ===\n")
    #         f.write(proc.stderr or "")

    #     # 如果返回码不为0，认为编译失败，清理并返回 0.0
    #     if proc.returncode != 0:
    #         # 你可以选择不删除 .tex/.log，让它们留着做进一步检查；下面仍然删除中间文件但保留 pdflatex_log_path
    #         for p in (pdf_path, aux_path, log_path, out_path):
    #             try:
    #                 if os.path.exists(p):
    #                     os.remove(p)
    #             except Exception:
    #                 pass
    #         # 保留 tex 和 pdflatex_log_path（便于手动调试），也可以选择删除 tex。
    #         return 0.0

    # except subprocess.TimeoutExpired as e:
    #     # 超时也要把已捕获的输出写到日志（TimeoutExpired 可能包含 stdout/stderr）
    #     import traceback
    #     with open(pdflatex_log_path, "w", encoding="utf-8") as f:
    #         f.write("=== pdflatex TIMEOUT after %s seconds ===\n\n" % 60)
    #         f.write("=== stdout (partial) ===\n")
    #         f.write((e.stdout or "") + "\n\n")
    #         f.write("=== stderr (partial) ===\n")
    #         f.write((e.stderr or "") + "\n\n")
    #         f.write("=== exception traceback ===\n")
    #         traceback.print_exc(file=f)
    #     # 清理中间产物（保留 tex 和 log 以便排查）
    #     for p in (pdf_path, aux_path, log_path, out_path):
    #         try:
    #             if os.path.exists(p):
    #                 os.remove(p)
    #         except Exception:
    #             pass
    #     import sys
    #     sys.exit(0)
    #     return 0.0
    if os.path.exists(tex_path):
        os.remove(tex_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(aux_path):
        os.remove(aux_path)
    if os.path.exists(log_path):
        os.remove(log_path)
    if os.path.exists(out_path):
        os.remove(out_path)
    return 1.0



def get_image(output_dir='pred_latex', image_filename='default.png'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.splitext(image_filename)[0]
    tex_path = os.path.join(output_dir, f"{base_name}.tex")
    pdf_path = tex_path.replace(".tex", ".pdf")
    pdf_path = pdf_path.replace("\\", "/")
    img_path = pdf_path.replace(".pdf", ".png")
    aux_path= pdf_path.replace(".pdf", ".aux")
    log_path= pdf_path.replace(".pdf", ".log")
    out_path= pdf_path.replace(".pdf", ".out")
    
    try:   
        # 2. 运行 pdfcrop 裁剪边缘，保留 5pt 边距
        subprocess.run(["pdfcrop","--margins", "5 5 5 5", pdf_path, pdf_path], #"--margins", "\"5 5 5 5\""
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"ERROR 2 pdfcrop fail: {pdf_path}")
        '''
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(aux_path):
            os.remove(aux_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        '''
        return False 
    try:   
        # 3. 运行 pdftoppm 转换为 PNG
        subprocess.run(["pdftoppm", "-png", "-r", "300", pdf_path, img_path], 
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.rename(f"{img_path}-1.png", img_path)
    except subprocess.CalledProcessError:
        print(f"ERROR 3 pdftoppm fail: {pdf_path}")
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


