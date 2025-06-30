import os

base_dir = "driver_dataset"
categories = ["drowsy", "yawning", "phone_call", "texting", "smoking", "looking_back", "normal"]

for folder in ["videos", "frames"]:
    for category in categories:
        os.makedirs(os.path.join(base_dir, folder, category), exist_ok=True)

os.makedirs(os.path.join(base_dir, "scripts"), exist_ok=True)
print("Folder structure created.")
