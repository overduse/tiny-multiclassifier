import os
from PIL import Image

def check_image_resolutions(root_dir, target_size=(32, 32)):
    """
    scan all the image files, check whether it contains the images
    have different image solution.

    Args:
        root_dir (str): root_path should be scaned ('./data/train')。
        target_size (tuple): expected image resolution (width, height)
    """
    print(f"starting iterate the dir: '{root_dir}'")
    print(f"Searching for the images whose resolution is not {target_size[0]}x{target_size[1]}")
    print("-" * 60)

    mismatched_files = []
    total_images_scanned = 0

    if not os.path.isdir(root_dir):
        print(f"Error: can not find the dir'{root_dir}'")
        return

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                total_images_scanned += 1
                file_path = os.path.join(dirpath, filename)
                
                try:
                    with Image.open(file_path) as img:
                        if img.size != target_size:
                            mismatched_files.append((file_path, img.size))
                except Exception as e:
                    print(f"can not read the file: {file_path} | Error: {e}")

    print("\n--- FINISH ---")
    if not mismatched_files:
        print(f"all of the {total_images_scanned} images are {target_size}。")
    else:
        print(f"within {total_images_scanned} images, we find that {len(mismatched_files)} images' resolution are wrong:")
        for path, size in mismatched_files:
            print(f"  - path: {path} | actual size: {size[0]}x{size[1]}")
    print("-" * 60)


if __name__ == '__main__':
    DATA_DIRECTORY = './data/train' 
    
    check_image_resolutions(root_dir=DATA_DIRECTORY, target_size=(32, 32))
