from flask import Blueprint, render_template
from .util import getParkingSpots

bp = Blueprint("pages", __name__)

bp.route("/")
def home():
    return render_template("pages/home.html")


@bp.route("/")
def home():
    numOfSpots = getParkingSpots()
    print(f"Number of spots: {numOfSpots}")
    return render_template("pages/home.html", numOfSpots=numOfSpots)
