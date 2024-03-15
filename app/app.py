from flask import Flask, Blueprint, render_template, url_for
from app.model.detect import ParkingSpaceDetector


def create_app():
    app = Flask(__name__)
    return app


app = create_app()

# Blueprint definition
bp = Blueprint("pages", __name__)


@bp.route("/")
def home():
    regions_path = "./model/resources/annotations.p"
    image_path = "./model/resources/parking-spaces.png"
    output_path = "./static/processed_image.png"

    detector = ParkingSpaceDetector(regions_path)

    _, empty_count = detector.process_image_and_highlight(image_path, output_path)

    img_url = url_for("static", filename="processed_image.png")

    print(f"Number of spots: {empty_count}")

    return render_template("home.html", numOfSpots=empty_count, imagePath=img_url)


# Registering blueprint
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
