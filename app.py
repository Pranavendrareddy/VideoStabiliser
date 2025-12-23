from flask import Flask, render_template, request, send_file, jsonify
from stabilise import convert_video, get_shaky_trajectory, smooth_trajectory, stabilise_video
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, no Tkinter
import matplotlib.pyplot as plt
import threading

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

Cx, Cy, Ca = None, None, None
smooth_Cx, smooth_Cy, smooth_Ca = None, None, None
progress = {"value": 0}


# -------------------- Plotting --------------------
def plotting(Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca, save_path="static/trajectory.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(Cx, label="x (shaky)")
    plt.plot(smooth_Cx, label="x (smooth)")
    plt.plot(Cy, label="y (shaky)")
    plt.plot(smooth_Cy, label="y (smooth)")
    plt.plot(np.degrees(Ca), label="angle (shaky)")
    plt.plot(np.degrees(smooth_Ca), label="angle (smooth)")
    plt.xlabel("Frame")
    plt.ylabel("Cumulative motion")
    plt.title("Camera trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# -------------------- Background stabilization --------------------
def run_stabilisation(input_path, output_path):
    global Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca, progress

    def progress_callback(val):
        progress["value"] = val

    # 0–50% progress: analysing trajectory
    Cx, Cy, Ca = get_shaky_trajectory(input_path, progress_callback=progress_callback)

    # 50–100% progress: stabilization
    smooth_Cx, smooth_Cy, smooth_Ca = smooth_trajectory(Cx, Cy, Ca)

    def stabilisation_progress(val):
        # scale 0–50% to 50–100%
        progress["value"] = 50 + int(val / 2)

    stabilise_video(Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca,
                    input_path, output_path, progress_callback=stabilisation_progress)

    progress["value"] = 100


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")  # See frontend code below


@app.route("/start", methods=["POST"])
def start():
    file = request.files["video"]
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    output_path = os.path.join(OUTPUT_FOLDER, "stabilized.mp4")

    global progress
    progress["value"] = 0

    threading.Thread(target=run_stabilisation, args=(input_path, output_path)).start()

    return jsonify(status="started")


@app.route("/progress")
def get_progress():
    return jsonify(progress)


@app.route("/download")
def download():
    output_path = os.path.join(OUTPUT_FOLDER, "stabilized.mp4")
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    return "No stabilized video yet!", 404


@app.route("/plot")
def plot_route():
    global Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca

    if Cx is None:
        return "No video processed yet!"

    img_path = plotting(Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca)
    return render_template("plot.html", img=img_path)


# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
