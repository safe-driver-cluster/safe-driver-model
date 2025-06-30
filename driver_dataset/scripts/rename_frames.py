import os

def rename_images(folder):
    files = sorted(os.listdir(folder))
    for i, f in enumerate(files):
        ext = os.path.splitext(f)[-1]
        os.rename(os.path.join(folder, f), os.path.join(folder, f"{os.path.basename(folder)}_{i}{ext}"))

if __name__ == "__main__":
    base_path = "../frames/"
    categories = ["drowsy", "yawning", "phone_call", "texting", "smoking", "looking_back", "normal"]
    for cat in categories:
        rename_images(os.path.join(base_path, cat))
