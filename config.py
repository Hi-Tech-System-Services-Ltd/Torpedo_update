from pathlib import Path
import zipfile

ROOT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = ROOT_DIR / "models"
MODEL_ZIP = ROOT_DIR / "model_files.zip"

def unzip_models():
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print("Unzipping model files...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
        print("Models unzipped successfully!")

# Call unzip function on startup
unzip_models()

# Create necessary directories structure
IMAGES_DIR = ROOT_DIR / "images"
TEMP_DIR = ROOT_DIR / "temp_img"
FLIR_IMAGES_DIR = ROOT_DIR.parent / "flir" / "images"

# Create directory structure
directories = {
    "images": {
        "image1": IMAGES_DIR / "image1",
        "image2": IMAGES_DIR / "image2",
        "image3": IMAGES_DIR / "image3"
    },
    "temp_img": {
        "temp_img1": TEMP_DIR / "temp_img1",
        "temp_img2": TEMP_DIR / "temp_img2",
        "temp_img3": TEMP_DIR / "temp_img3"
    }
}

# Create all required directories
def create_directory_structure():
    for category, subdirs in directories.items():
        for subdir_name, path in subdirs.items():
            path.mkdir(parents=True, exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FLIR_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Image paths
THERMAL_IMAGES = {
    "cam1": FLIR_IMAGES_DIR / "thermal_image_1.jpg",
    "cam2": FLIR_IMAGES_DIR / "thermal_image_2.jpg",
    "cam3": FLIR_IMAGES_DIR / "thermal_image_3.jpg"
}

# Model paths
MODEL_PATHS = {
    "cam1": {
        "classify": MODELS_DIR / "cam1_classify.pt",
        "detect": MODELS_DIR / "cam1_detect.pt"
    },
    "cam2": {
        "classify": MODELS_DIR / "cam2_classify.pt",
        "detect": MODELS_DIR / "cam2_detect.pt"
    },
    "cam3": {
        "classify": MODELS_DIR / "cam3_classify.pt",
        "detect": MODELS_DIR / "cam3_detect.pt"
    }
}

# Initialize directory structure
create_directory_structure()
