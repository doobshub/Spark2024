from flask import Blueprint, render_template
from .util import getParkingSpots
from detect import ParkingSpaceDetector

bp = Blueprint("pages", __name__)

bp.route("/")


def home():
    return render_template("pages/home.html")


def getParkingSpots():
    regions_path = "presentationRegion.p"
    image_path = "presentationImage.png"

    detector = ParkingSpaceDetector(regions_path)

    _, empty_count = detector.process_image_and_highlight(image_path)

    return empty_count


@bp.route("/")
def home():
    numOfSpots = getParkingSpots()
    print(f"Number of spots: {numOfSpots}")
    return render_template("pages/home.html", numOfSpots=numOfSpots)
