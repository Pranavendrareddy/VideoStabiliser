from flask import Flask, render_template, request, send_file
from stabilise import convert_video, get_shaky_trajectory, smooth_trajectory, stabilise_video
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["video"]
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        converted = convert_video(input_path)
        Cx, Cy, Ca = get_shaky_trajectory(converted)
        smooth_Cx, smooth_Cy, smooth_Ca = smooth_trajectory(Cx, Cy, Ca)

        output_path = os.path.join(OUTPUT_FOLDER, "stabilized.mp4")
        stabilise_video(
            converted, Cx, Cy, Ca,
            smooth_Cx, smooth_Cy, smooth_Ca,
            output_path
        )

        return send_file(output_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
