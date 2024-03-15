import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
import numpy as np
import pickle
from PIL import Image


class ParkingSpaceDetector:
    """
    A class for detecting empty parking spaces in an image.
    """

    def __init__(
        self, regions_path, model_weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ):
        """
        Initializes the detector with the path to the regions file and loads the model.

        Args:
            regions_path (str): Path to the pickle file containing parking space regions.
            model_weights: Pre-trained weights for the detection model.
        """
        self.regions = self.load_regions(regions_path)
        self.model = self.load_model(model_weights)

    @staticmethod
    def load_regions(regions_path):
        """
        Loads regions of interest from a pickle file.

        Args:
            regions_path (str): Path to the pickle file.

        Returns:
            list: A list of regions (each defined by four points).
        """
        with open(regions_path, "rb") as file:
            regions = pickle.load(file)
        return regions

    @staticmethod
    def load_model(model_weights):
        """
        Loads the pre-trained SSD MobileNetV3-Large model.

        Args:
            model_weights: Pre-trained weights for the model.

        Returns:
            model: The loaded and initialized model.
        """
        model = ssdlite320_mobilenet_v3_large(weights=model_weights)
        model.eval()
        return model

    def process_image_and_highlight(self, image_path, threshold=0.031):
        """
        Processes an image to detect and highlight empty parking spaces.

        Args:
            image_path (str): Path to the image file.
            threshold (float): Overlap threshold to determine if a space is empty.

        Returns:
            tuple: A tuple containing the processed image and the count of empty spaces.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image at {image_path}")
            return None, 0

        image_tensor = self.transform_image(image)
        car_boxes = self.detect_cars(image_tensor)
        overlaps = self.check_region_overlap(car_boxes, threshold)
        empty_count = self.highlight_empty_spaces(image, overlaps)

        return image, empty_count

    @staticmethod
    def transform_image(image):
        """
        Transforms an image for model processing.

        Args:
            image (numpy.ndarray): The image to transform.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = T.Compose([T.ToTensor()])
        return transform(image_pil)

    def detect_cars(self, image_tensor):
        """
        Detects cars in the given image tensor.

        Args:
            image_tensor (torch.Tensor): The image tensor.

        Returns:
            list: A list of detected car bounding boxes.
        """
        with torch.no_grad():
            prediction = self.model([image_tensor])
        car_boxes = [
            box.tolist()
            for box, label in zip(prediction[0]["boxes"], prediction[0]["labels"])
            if label.item() == 3
        ]
        return car_boxes

    def check_region_overlap(self, boxes, threshold):
        """
        Checks for overlap between regions and detected bounding boxes.

        Args:
            boxes (list): List of detected car bounding boxes.
            threshold (float): Overlap threshold to consider a space filled.

        Returns:
            list: A list indicating whether each region is empty (True for empty).
        """
        overlaps = [True] * len(self.regions)  # Assume all regions are empty initially
        for i, region in enumerate(self.regions):
            region_bbox = self.quadrilateral_to_bbox(region)
            region_area = self.rect_intersection_area(region_bbox, region_bbox)
            for box in boxes:
                box_area = self.rect_intersection_area(box, box)
                overlap_area = self.rect_intersection_area(region_bbox, box)
                if overlap_area / box_area > threshold:
                    overlaps[i] = False
                    break
        return overlaps

    @staticmethod
    def quadrilateral_to_bbox(quadrilateral):
        """
        Converts a quadrilateral defined by four points into a bounding box.
        """
        xs = [point[0] for point in quadrilateral]
        ys = [point[1] for point in quadrilateral]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def rect_intersection_area(a, b):
        """
        Calculates the area of intersection between two rectangles.
        """
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if dx >= 0 and dy >= 0:
            return dx * dy
        return 0

    def highlight_empty_spaces(self, image, overlaps):
        """
        Highlights empty parking spaces on the image.
        """
        empty_count = 0
        for i, region in enumerate(self.regions):
            if overlaps[i]:  # If true, the space is considered empty
                bbox = self.quadrilateral_to_bbox(region)
                cv2.rectangle(
                    image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )
                empty_count += 1
        return empty_count


# Example usage
detector = ParkingSpaceDetector("presentationRegion.p")
image_path = "presentationImage.png"
processed_image, empty_count = detector.process_image_and_highlight(image_path)

if processed_image is not None:
    cv2.imshow("Empty Spaces Highlighted", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Number of empty spaces: {empty_count}")
else:
    print("Failed to process the image.")
