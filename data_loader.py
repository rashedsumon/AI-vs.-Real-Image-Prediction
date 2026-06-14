import os
import kagglehub

def get_dataset_path():
    """
    Automatically downloads the latest version of the CIFAKE dataset 
    via kagglehub and returns the absolute directory path.
    """
    try:
        # Downloads the dataset if not present; returns local path if already cached.
        path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
        return path
    except Exception as e:
        print(f"Error downloading dataset via kagglehub: {e}")
        return None

if __name__ == "__main__":
    # Test block to verify download paths locally
    dataset_dir = get_dataset_path()
    if dataset_dir:
        print("Path to dataset files successfully resolved:", dataset_dir)
        print("Contents:", os.listdir(dataset_dir))