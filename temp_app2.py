import pyodbc
import datetime
from ultralytics import YOLO
import os
import shutil
import time
import threading
import requests
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='detection_log.txt'
)

# Load environment variables
load_dotenv()

# Directories
save_directory = "images\\image3"  # Directory for saving images
temp_image_dir = "temp_img\\temp_img3"  # Directory for temporary image storage

# Ensure directories exist
os.makedirs(save_directory, exist_ok=True)
os.makedirs(temp_image_dir, exist_ok=True)

# Image paths
cam1_image_path = "E:\\flir\\images\\thermal_image_3.jpg"

# Model paths
torpedo_classification_model_path = "E:\\flir\\Torpedo_Detection\\models\\cam3_classify.pt"
cam1_detection_model_path = "E:\\flir\\Torpedo_Detection\\models\\cam3_detect.pt"

# Load models
torpedo_classification_model = YOLO(torpedo_classification_model_path)
cam1_detection_model = YOLO(cam1_detection_model_path)

def create_requests_session(retries=3, backoff_factor=0.3):
    """
    Create a requests session with retry mechanism.

    Args:
        retries (int): Number of retry attempts
        backoff_factor (float): Backoff factor for exponential backoff

    Returns:
        requests.Session: Configured requests session
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Class to detect the largest bounding box
class LargestBoundingBoxDetector:
    def __init__(self, model, save_directory, temp_image_dir):
        self.model = model
        self.save_directory = save_directory
        self.temp_image_dir = temp_image_dir
        self.largest_box = None
        self.largest_area = 0
        self.save_path = os.path.join(self.temp_image_dir, "largest_box_image.jpg")

    def save_image(self, source_image_path):
        """Save the image to a temporary directory."""
        try:
            logging.info(f"Saving image to {self.save_path}")
            shutil.copy(source_image_path, self.save_path)
        except Exception as e:
            logging.error(f"Error saving image: {e}")

    def detect_objects(self, image_path):
        """
        Detect objects and find the largest bounding box.

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: Largest bounding box coordinates and dimensions
        """
        try:
            results = self.model(image_path)
            if not results:
                logging.warning("No results returned from model.")
                return None

            for result in results:
                boxes = result.boxes
                if not boxes:
                    logging.warning("No boxes detected.")
                    return None

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().detach().numpy().astype(int)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    if area > self.largest_area:
                        self.largest_box = (x1, y1, x2, y2, width, height)
                        self.largest_area = area
                        self.save_image(image_path)

            return self.largest_box

        except Exception as e:
            logging.error(f"Error in object detection: {e}")
            return None

    def compare_and_update(self, image_path):
        """
        Compare current detection with previous largest box.

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: Updated largest bounding box
        """
        current_box = self.detect_objects(image_path)
        if current_box:
            x1, y1, x2, y2, width, height = current_box
            current_area = width * height

            if current_area < self.largest_area:
                logging.info(f"Reducing size detected. Returning last largest bounding box: {self.largest_box}")
                return self.largest_box

        return current_box

def calculate_bbox_info(x1, y1, x2, y2):
    """
    Calculate the center coordinates of a bounding box.

    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates

    Returns:
        tuple: Center x and y coordinates
    """
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def insert_detection_data(torpedo_id, c_x, c_y, w, h, cam_id, filename):
    """
    Send detection data to the API endpoint.

    Args:
        torpedo_id (int): Torpedo identifier
        c_x, c_y (int): Center coordinates
        w, h (int): Width and height
        cam_id (int): Camera identifier
        filename (str): Image filename
    """
    # Get API URL from environment or use default
    api_url = os.getenv('API_ENDPOINT', 'http://localhost:51655//API/detectPush')

    data = {
        "TorpedoID": int(torpedo_id),
        "centerx": int(c_x),
        "centery": int(c_y),
        "width": int(w),
        "height": int(h),
        "udt": 0 if 200 < c_x < 400 else 2,
        "cameraID": int(cam_id),
        "filename": filename
    }

    try:
        # Create a session with retry mechanism
        session = create_requests_session()

        # Send POST request
        response = session.post(api_url, json=data, timeout=10)
        response.raise_for_status()

        logging.info("Successfully pushed detection data")
        print("Detection data pushed successfully")

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        print(f"API request failed: {e}")

def is_torpedo_present():
    """
    Check if a torpedo is present in the frame.

    Returns:
        bool: True if torpedo is detected, False otherwise
    """
    results = torpedo_classification_model.predict(source=cam1_image_path)

    if not results:
        logging.warning("No results from torpedo model.")
        return False

    for result in results:
        top1_class_id = result.probs.top1
        top1_confidence = result.probs.top1conf
        s = f"{top1_class_id} {result.names[top1_class_id]} {top1_confidence:.2f}"

        if "NonTorpedoFrame" in s:
            return False

    return True

def main():
    """
    Main function to start torpedo detection process.
    """
    detector_cam1 = LargestBoundingBoxDetector(cam1_detection_model, save_directory, temp_image_dir)
    torpedo_id = 100

    def cam1_detection_thread():
        """
        Thread function for camera 1 detection process.
        """
        coordinates_cam1 = detector_cam1.detect_objects(cam1_image_path)

        if coordinates_cam1 is None:
            logging.warning("No bounding box detected, skipping data push.")
            print("No bounding box detected, skipping data push.")
            return

        x1_cam1, y1_cam1, x2_cam1, y2_cam1, width_cam1, height_cam1 = coordinates_cam1
        center_x, center_y = calculate_bbox_info(x1_cam1, y1_cam1, x2_cam1, y2_cam1)

        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.jpg"
        save_image_path = os.path.join(save_directory, filename)

        try:
            shutil.copy(
                os.path.join(temp_image_dir, "largest_box_image.jpg"),
                save_image_path
            )
        except FileNotFoundError as e:
            logging.error(f"File not found error: {e}")
            print(f"File not found error: {e}")

        insert_detection_data(torpedo_id, center_x, center_y, width_cam1, height_cam1, 2, filename)
        print("Cam2 detection process completed\n")

    torpedo_in_frame = is_torpedo_present()

    if torpedo_in_frame:
        cam2_thread = threading.Thread(target=cam1_detection_thread)
        cam2_thread.start()
        cam2_thread.join()
    else:
        logging.info("No torpedo detected, hence not pushed")
        print("No torpedo detected, hence not pushed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {e}")
        print(f"Unexpected error: {e}")
