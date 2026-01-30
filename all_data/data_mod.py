import os
import zipfile

root_dir = "/lus/flare/projects/Prob_AI/kanadsen/all_data/ARO_data/dataset_trial2"        # Directory you want to scan (recursively)
extract_root = "/lus/flare/projects/Prob_AI/kanadsen/all_data/ARO_data/dataset_trial2"   # Where extracted folders will be created

os.makedirs(extract_root, exist_ok=True)

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".zip"):
            zip_path = os.path.join(dirpath, filename)
            extract_folder = os.path.join(
                extract_root, 
                os.path.relpath(dirpath, root_dir),   # Keep folder structure
                filename.replace(".zip", "")
            )
            
            os.makedirs(extract_folder, exist_ok=True)
            print(f"Extracting {zip_path} -> {extract_folder}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)

print("Finished extracting all zip files.")
