from flask import Blueprint, render_template, url_for
from app.model.detect import ParkingSpaceDetector

bp = Blueprint("pages", __name__)

bp.route("/")


def getParkingSpots():
    regions_path = "./model/resources/annotations.p"
    image_path = "./model/resources/parking-spaces.png"
    output_path = "static/processed_image.png"

    detector = ParkingSpaceDetector(regions_path)

    _, empty_count = detector.process_image_and_highlight(image_path, output_path)

    return empty_count, url_for("static", filename="processed_image.png")


def home():
    return render_template("pages/home.html")


@bp.route("/")
def home():
    numOfSpots, imagePath = getParkingSpots()
    print(f"Number of spots: {numOfSpots}")
    return render_template(
        "pages/home.html", numOfSpots=numOfSpots, imagePath=imagePath
    )
