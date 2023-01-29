import cv2
from flask import Flask, render_template, url_for
import camtestRef
import fakeGen
"""
import matplotlib
matplotlib.use("Agg")
"""
app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def hell():
    return render_template("html_practice.html")

@app.route("/camera")
def camtest():
    camtestRef.main()
    return render_template("takephoto.html")

@app.route("/fakeimage")
def fake():
    fakeGen.main()
    return render_template("show_fake.html")

@app.route("/showpicture")
def showpicture():
    return render_template("show_picture.html")




"""
@app.route('/take-photo')
def take_photo():
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        cv2.imshow("you", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('c'):
            cv2.imwrite("./templates/camera_capture.jpg", frame)
        elif key == ord('q'):
            break
        
    return "taking a photo"
"""

if __name__ == "__main__":
    app.run(debug=True)