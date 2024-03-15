import os
import numpy as np
import cv2
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection


class RegionSelector:
    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.poly_selector = PolygonSelector(ax, self.on_select)
        self.selected_vertices = []

    def on_select(self, vertices):
        self.selected_vertices = vertices
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly_selector.disconnect_events()
        self.canvas.draw_idle()


def save_regions_and_exit(event):
    global region_selector, save_path, regions
    if event.key == "b":
        region_selector.disconnect()
        if os.path.exists(save_path):
            os.remove(save_path)
        print("Regions saved in:", save_path)
        with open(save_path, "wb") as f:
            pickle.dump(regions, f, protocol=pickle.HIGHEST_PROTOCOL)
        plt.close()


def main():
    global region_selector, save_path, regions

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path of video file")
    parser.add_argument(
        "--out_file", help="Name of the output file", default="regions.p"
    )
    args = parser.parse_args()
    save_path = args.out_file if args.out_file.endswith(".p") else args.out_file + ".p"

    # Print instructions
    print("Instructions:")
    print("1. Select a region in the figure by enclosing them within a quadrilateral.")
    print("2. Press 'f' key to go full screen.")
    print("3. Press 'esc' key to discard current quadrilateral.")
    print("4. Try holding the 'shift' key to move all of the vertices.")
    print("5. Try holding the 'ctrl' key to move a single vertex.")
    print("6. After marking a quadrilateral press 'n' to save current quadrilateral.")
    print("7. Press 'q' to start marking a new quadrilateral.")
    print("8. Press 'b' when done to exit the program.\n")

    # Video capture to get an image
    video_capture = cv2.VideoCapture(args.video_path)
    success, frame = video_capture.read()
    if not success:
        print("Error: Unable to read the video file.")
        return
    rgb_image = frame[:, :, ::-1]
    video_capture.release()

    regions = []  # Store selected regions

    while True:
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)

        patches = [Polygon(region) for region in regions]
        p = PatchCollection(patches, alpha=0.7)
        p.set_array(10 * np.ones(len(patches)))
        ax.add_collection(p)

        region_selector = RegionSelector(ax)

        # Event connections
        plt.connect("key_press_event", save_regions_and_exit)
        plt.show()

        if len(region_selector.selected_vertices) == 4:
            regions.append(region_selector.selected_vertices)


if __name__ == "__main__":
    main()
