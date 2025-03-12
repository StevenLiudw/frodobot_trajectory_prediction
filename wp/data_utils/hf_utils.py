import os
import zipfile
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from huggingface_hub import HfApi, upload_folder, snapshot_download
import dotenv

dotenv.load_dotenv()

def zip_subset_folders(base_dir: str = "out/sampled_frames_1/") -> List[str]:
    """
    Zip each subset folder in the base directory.
    
    Args:
        base_dir: Directory containing subset folders
        
    Returns:
        List of paths to created zip files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory {base_dir} does not exist")
    
    zip_files = []
    
    # Find all subset folders
    subset_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("subset_")]
    
    for folder in subset_folders:
        zip_path = folder.with_suffix('.zip')
        print(f"Creating {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, base_path)
                    zipf.write(file_path, arcname)
        
        zip_files.append(str(zip_path))
    
    return zip_files


def upload_dataset(
    zip_files: List[str], 
    repo_id: str, 
    token: Optional[str] = None,
) -> None:
    """
    Upload zip files to Hugging Face dataset repository.
    
    Args:
        zip_files: List of zip file paths to upload
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')
        token: Hugging Face API token
    """
    if not token:
        print("Warning: No Hugging Face token provided. Using environment variable if available.")
    
    api = HfApi(token=token)
    
    # Create a temporary directory to organize files
    tmp_dir = Path("tmp_upload_dir")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    
    # Copy zip files to temporary directory, preserving parent folder structure if needed
    for zip_file in zip_files:
        zip_path = Path(zip_file)
        
        # Extract parent folder name from the path
        parent_folder = zip_path.parent.name
        # Create parent folder in tmp_dir if it doesn't exist
        parent_dir = tmp_dir / parent_folder
        parent_dir.mkdir(exist_ok=True)
        # Copy zip file to parent folder in tmp_dir
        shutil.copy(zip_file, parent_dir)
    
    # Upload the directory
    print(f"Uploading dataset to {repo_id}...")
    upload_folder(
        folder_path=str(tmp_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    # Clean up
    shutil.rmtree(tmp_dir)
    print("Upload complete!")


def download_and_unzip_dataset(
    repo_id: str, 
    output_dir: str = "out/sampled_frames_1/",
    token: Optional[str] = None
) -> None:
    """
    Download a dataset from Hugging Face and unzip its contents.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')
        output_dir: Directory where the dataset will be extracted
        token: Hugging Face API token
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download the dataset
    print(f"Downloading dataset from {repo_id}...")
    download_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    
    # Find and extract all zip files
    download_path = Path(download_dir)
    zip_files = list(download_path.glob("*.zip"))
    
    for zip_file in zip_files:
        subset_name = zip_file.stem
        subset_dir = output_path / subset_name
        
        print(f"Extracting {zip_file} to {subset_dir}...")
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(output_path)
    
    print(f"Dataset downloaded and extracted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Zip and upload dataset to Hugging Face")
    parser.add_argument("--base_dir", default="out/sampled_frames_1/", 
                        help="Directory containing subset folders")
    parser.add_argument("--repo_id", default="jamiewjm/fd",
                        help="Hugging Face repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", default=os.getenv("HF_API_TOKEN"), 
                        help="Hugging Face API token")
    parser.add_argument("--upload", action="store_true", 
                        help="Upload dataset to Hugging Face")
    parser.add_argument("--download", action="store_true", 
                        help="Download dataset from Hugging Face")
    parser.add_argument("--output_dir", default="out/sampled_frames_1/", 
                        help="Directory to extract downloaded dataset")
    
    args = parser.parse_args()
    
    if args.upload:
        zip_files = zip_subset_folders(args.base_dir)
        if zip_files:
            upload_dataset(zip_files, args.repo_id, args.token)
    
    if args.download:
        download_and_unzip_dataset(args.repo_id, args.output_dir, args.token)


if __name__ == "__main__":
    main()
