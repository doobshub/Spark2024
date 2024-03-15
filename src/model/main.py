import os
import cv2
import numpy as np
import pickle
import argparse
from shapely.geometry import Polygon as shapely_poly
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn.utils import download_trained_weights


class ParkingLotConfig(Config):
    NAME = "parking_lot_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81


class ParkingLotDetector:
    def __init__(self, model_weights_path, regions_path):
        self.model = self.load_model(model_weights_path)
        self.parked_car_boxes = self.load_parked_car_boxes(regions_path)

    def load_model(self, weights_path):
        model = MaskRCNN(
            mode="inference",
            model_dir=os.path.join(os.getcwd(), "logs"),
            config=ParkingLotConfig(),
        )
        if not os.path.exists(weights_path):
            download_trained_weights(weights_path)
        model.load_weights(weights_path, by_name=True)
        return model

    def load_parked_car_boxes(self, regions_path):
        with open(regions_path, "rb") as f:
            return pickle.load(f)

    def detect_cars(self, frame):
        rgb_image = frame[:, :, ::-1]
        results = self.model.detect([rgb_image], verbose=0)
        return results[0]["rois"], results[0]["class_ids"]

    def compute_overlaps(self, car_boxes):
        new_car_boxes = []
        for box in car_boxes:
            y1, x1, y2, x2 = box
            p1, p2, p3, p4 = (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            new_car_boxes.append([p1, p2, p3, p4])

        overlaps = np.zeros((len(self.parked_car_boxes), len(new_car_boxes)))
        for i, parking_area in enumerate(self.parked_car_boxes):
            for j, box in enumerate(new_car_boxes):
                polygon1_shape = shapely_poly(parking_area)
                polygon2_shape = shapely_poly(box)

                intersection = polygon1_shape.intersection(polygon2_shape).area
                union = polygon1_shape.union(polygon2_shape).area
                overlaps[i][j] = intersection / union
        return overlaps

    def mark_free_spaces(self, frame, alpha=0.6, threshold=0.15):
        cars_boxes, class_ids = self.detect_cars(frame)
        cars = [box for i, box in enumerate(cars_boxes) if class_ids[i] in [3, 8, 6]]
        overlaps = self.compute_overlaps(cars)

        overlay = frame.copy()
        for parking_area, overlap_areas in zip(self.parked_car_boxes, overlaps):
            max_overlap = np.max(overlap_areas)
            if max_overlap < threshold:
                cv2.fillPoly(overlay, [np.array(parking_area)], (71, 27, 92))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Video file")
    parser.add_argument("regions_path", help="Regions file", default="regions.p")
    args = parser.parse_args()

    detector = ParkingLotDetector("mask_rcnn_coco.h5", args.regions_path)

    video_capture = cv2.VideoCapture(args.video_path)
    out = cv2.VideoWriter(
        "out.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        video_capture.get(cv2.CAP_PROP_FPS),
        (
            int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        marked_frame = detector.mark_free_spaces(frame)
        cv2.imshow("output", marked_frame)
        out.write(marked_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("Output saved as out.avi")
