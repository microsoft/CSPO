import os
import shutil

def process_image_name(root_dir):
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist.")
    else:
        print(f"Success: Directory {root_dir} exists.")
    # 遍历所有子目录和文件
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".png-1.png"):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(".png-1.png", ".png")
                new_path = os.path.join(dirpath, new_filename)

                # 重命名文件
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

def process_path(input_base_dir):
    # Step 1: Rename 'image' folder to '1'
    old_image_path = os.path.join(input_base_dir, "image")
    new_dir_path = os.path.join(input_base_dir, "1")
    if os.path.exists(old_image_path):
        os.rename(old_image_path, new_dir_path)

    # Step 2: Remove '1/image' if it exists
    old_nested_image = os.path.join(new_dir_path, "image")
    if os.path.exists(old_nested_image):
        shutil.rmtree(old_nested_image)

    # Step 3: Rename '1/table_tex_file' to '1/image'
    old_tex_path = os.path.join(new_dir_path, "table_tex_file")
    new_image_path = os.path.join(new_dir_path, "image")
    if os.path.exists(old_tex_path):
        os.rename(old_tex_path, new_image_path)

    # Step 4: Move '1/image' to 'econ_2020'
    final_image_path = os.path.join(input_base_dir, "image")
    if os.path.exists(new_image_path):
        shutil.move(new_image_path, final_image_path)

    process_image_name(final_image_path)

    # Step 5: Delete '1' directory
    if os.path.exists(new_dir_path):
        os.rmdir(new_dir_path)
