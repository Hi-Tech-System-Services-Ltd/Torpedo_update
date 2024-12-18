import datetime
import os
import shutil
import threading
import requests
import logging
from dotenv import load_dotenv
from ultralytics import YOLO
from pathlib import Path
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='detection_log.txt'
)

# Load environment variables
load_dotenv()

class TorpedoDetectionTracker:
    def __init__(self, detection_interval=300):
        self.last_detection_time = None
        self.detection_interval = detection_interval
        self.first_detection_data = None

    def can_record_detection(self):
        current_time = datetime.datetime.now()
        if self.last_detection_time is None:
            return True
        time_diff = (current_time - self.last_detection_time).total_seconds()
        return time_diff > self.detection_interval

    def record_detection(self, detection_data):
        if self.can_record_detection():
            self.first_detection_data = detection_data
            self.last_detection_time = datetime.datetime.now()
            return True
        return False

def create_requests_session(retries=3, backoff_factor=0.3):
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

class LargestBoundingBoxDetector:
    def __init__(self, model, save_directory, temp_image_dir):
        self.model = model
        self.save_directory = save_directory
        self.temp_image_dir = temp_image_dir
        self.largest_box = None
        self.largest_area = 0
        self.save_path = os.path.join(self.temp_image_dir, "largest_box_image.jpg")

    def save_image(self, source_image_path):
        try:
            logging.info(f"Saving image to {self.save_path}")
            shutil.copy(source_image_path, self.save_path)
        except Exception as e:
            logging.error(f"Error saving image: {e}")

    def detect_objects(self, image_path):
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

def calculate_bbox_info(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def insert_detection_data(torpedo_tracker, torpedo_id, c_x, c_y, w, h, cam_id, filename):
    detection_data = {
        "TorpedoID": int(torpedo_id),
        "centerx": int(c_x),
        "centery": int(c_y),
        "width": int(w),
        "height": int(h),
        "udt": 0 if 200 < c_x < 400 else 2,
        "cameraID": int(cam_id),
        "filename": filename
    }

    if torpedo_tracker.record_detection(detection_data):
        api_url = os.getenv('API_ENDPOINT', 'http://192.168.1.1:8080//API/detectPush')

        try:
            session = create_requests_session()
            response = session.post(api_url, json=detection_data, timeout=10)
            response.raise_for_status()

            logging.info("Successfully pushed detection data")
            print("Detection data pushed successfully")

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            print(f"API request failed: {e}")
    else:
        logging.info("Skipping detection - already recorded within time window")
        print("Skipping detection - already recorded within time window")

def is_torpedo_present(classification_model, image_path):
    results = classification_model.predict(source=image_path)

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

def main(cam_id, image_path, classify_model_path, detect_model_path, save_directory, temp_image_dir):
    classification_model = YOLO(classify_model_path)
    detection_model = YOLO(detect_model_path)
    detector = LargestBoundingBoxDetector(detection_model, save_directory, temp_image_dir)
    torpedo_id = 100
    torpedo_tracker = TorpedoDetectionTracker(detection_interval=300)

    def detection_thread():
        coordinates = detector.detect_objects(image_path)

        if coordinates is None:
            logging.warning("No bounding box detected, skipping data push.")
            print("No bounding box detected, skipping data push.")
            return

        x1, y1, x2, y2, width, height = coordinates
        center_x, center_y = calculate_bbox_info(x1, y1, x2, y2)

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

        insert_detection_data(
            torpedo_tracker,
            torpedo_id,
            center_x,
            center_y,
            width,
            height,
            cam_id,
            filename
        )
        print(f"Cam{cam_id} detection process completed\n")

    if is_torpedo_present(classification_model, image_path):
        detection_thread = threading.Thread(target=detection_thread)
        detection_thread.start()
        detection_thread.join()
    else:
        logging.info("No torpedo detected, hence not pushed")
        print("No torpedo detected, hence not pushed")

if __name__ == "__main__":
    try:
        import sys
        cam_id = int(sys.argv[1])
        image_path = config.THERMAL_IMAGES[f"cam{cam_id}"]
        classify_model_path = config.MODEL_PATHS[f"cam{cam_id}"]["classify"]
        detect_model_path = config.MODEL_PATHS[f"cam{cam_id}"]["detect"]
        save_directory = config.directories["images"][f"image{cam_id}"]
        temp_image_dir = config.directories["temp_img"][f"temp_img{cam_id}"]
        main(cam_id, image_path, classify_model_path, detect_model_path, save_directory, temp_image_dir)
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {e}")
        print(f"Unexpected error: {e}")
