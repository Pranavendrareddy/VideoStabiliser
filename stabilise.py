import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scipy.signal import medfilt

import subprocess
import os

def fix_border(frame, scale=1.08):
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
    return cv2.warpAffine(frame, T, (w, h))

def fix_border_to_stable2(frame, dx, dy, da, scale=1.1):
    h, w = frame.shape[:2]
    center_x = w/2 + dx   # shift the center by the translation
    center_y = h/2 + dy
    T = cv2.getRotationMatrix2D((center_x, center_y), da, scale)
    return cv2.warpAffine(frame, T, (w, h))

def fix_border_to_stable(frame, dx, dy, da, scale=1.1):

    h, w = frame.shape[:2]

    # Convert rotation to degrees
    angle_deg = np.degrees(da)

    # Compute the center of the frame around which to rotate
    # Shift the rotation center so the "stable point" is at the middle
    center = (w / 2 + dx, h / 2 + dy)

    # Construct rotation + scale matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    # Apply affine transformation
    stabilized = cv2.warpAffine(
        frame, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # optionally use BORDER_REFLECT for smoother edges
    )

    # Optional: crop extra edges to fully remove borders
    # Compute crop region after scaling
    crop_w = int(w / scale)
    crop_h = int(h / scale)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    stabilized = stabilized[y1:y1+crop_h, x1:x1+crop_w]

    # Resize back to original frame size
    stabilized = cv2.resize(stabilized, (w, h))

    return stabilized

def convert_video(input_path, output_path="converted.mp4"):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-r", "30",
        "-pix_fmt", "yuv420p",
        "-vcodec", "libx264",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def get_shaky_trajectory(video_path = "video.mp4", progress_callback=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=total_frames, desc="Analysing motion")

    frame_counter = 0

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Empty video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 30, blockSize=3)

    Cx, Cy, Ca = [0.0], [0.0], [0.0]

    pbar = tqdm(total=total_frames-1, desc="Analysing motion")


    while True:

        ret, frame = cap.read()
        if ret:
            print("Frame shape:", frame.shape)
        else:
            print("Failed to read frame")
        if not ret:
            break

        # frame_with_corners = prev_frame.copy()
        # for pt in prev_points:
        #     x, y = pt.ravel()
        #     cv2.circle(frame_with_corners, (int(x), int(y)), 3, (0, 0, 255), -1)
        #
        # cv2.imshow("Corners", frame_with_corners)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

        valid_prev = prev_points[status.flatten() == 1]
        valid_curr = curr_points[status.flatten() == 1]

        M, inliers = cv2.estimateAffinePartial2D(
            valid_prev,
            valid_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            confidence=0.99
        )

        if M is None:
            dx = dy = da = 0
        else:
            dx = M[0, 2]
            dy = M[1, 2]
            da = np.arctan2(M[1, 0], M[0, 0])

        Cx.append(Cx[-1] + dx)
        Cy.append(Cy[-1] + dy)
        Ca.append(Ca[-1] + da)

        prev_gray = gray
        prev_points = valid_curr.reshape(-1, 1, 2)

        if len(prev_points) < 80:
            prev_points = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3
            )

        frame_counter += 1
        pbar.update(1)
        if progress_callback:
            # 50% of progress is assigned to analysis
            progress_callback(int(50 * frame_counter / total_frames))

    # plt.figure(figsize=(10, 4))
    # plt.plot(Cx, label="x")
    # plt.plot(Cy, label="y")
    # plt.plot(np.degrees(Ca), label="angle (deg)")
    # plt.legend()
    # plt.xlabel("Frame")
    # plt.ylabel("Cumulative motion")
    # plt.title("Shaky camera trajectory")
    # plt.show()

    pbar.close()
    cap.release()

    return Cx, Cy, Ca


def smooth_trajectory(Cx, Cy, Ca, max_trans=200, max_rot = 0.5):

    n = len(Cx)

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError("Could not create solver GLOP")

    Px = [solver.NumVar(-1e6, 1e6, f'Px_{i}') for i in range(n)]
    Py = [solver.NumVar(-1e6, 1e6, f'Py_{i}') for i in range(n)]
    Pa = [solver.NumVar(-1e6, 1e6, f'Pa_{i}') for i in range(n)]

    for i in range(3):
        solver.Add(Px[i] == Cx[i])
        solver.Add(Py[i] == Cy[i])
        solver.Add(Pa[i] == Ca[i])
    for i in range(n-3, n):
        solver.Add(Px[i] == Cx[i])
        solver.Add(Py[i] == Cy[i])
        solver.Add(Pa[i] == Ca[i])

    for i in range(n):
        solver.Add(Px[i] <= Cx[i] + max_trans)
        solver.Add(Px[i] >= Cx[i] - max_trans)
        solver.Add(Py[i] <= Cy[i] + max_trans)
        solver.Add(Py[i] >= Cy[i] - max_trans)
        solver.Add(Pa[i] <= Ca[i] + np.radians(max_rot))
        solver.Add(Pa[i] >= Ca[i] - np.radians(max_rot))

    Vx = [solver.NumVar(-1e6, 1e6, f'Vx_{i}') for i in range(n - 1)]
    Vy = [solver.NumVar(-1e6, 1e6, f'Vy_{i}') for i in range(n - 1)]
    Va = [solver.NumVar(-1e6, 1e6, f'Va_{i}') for i in range(n - 1)]

    for i in range(n - 1):
        dx = Px[i + 1] - Px[i]
        dy = Py[i + 1] - Py[i]
        da = Pa[i + 1] - Pa[i]

        solver.Add(Vx[i] >= dx)
        solver.Add(Vx[i] >= -dx)

        solver.Add(Vy[i] >= dy)
        solver.Add(Vy[i] >= -dy)

        solver.Add(Va[i] >= da)
        solver.Add(Va[i] >= -da)

    Ax = [solver.NumVar(-1e6, 1e6, f'Ax_{i}') for i in range(n - 2)]
    Ay = [solver.NumVar(-1e6, 1e6, f'Ay_{i}') for i in range(n - 2)]
    Aa = [solver.NumVar(-1e6, 1e6, f'Aa_{i}') for i in range(n - 2)]

    for i in range(1, n - 1):
        ax = Px[i + 1] - 2 * Px[i] + Px[i - 1]
        ay = Py[i + 1] - 2 * Py[i] + Py[i - 1]
        aa = Pa[i + 1] - 2 * Pa[i] + Pa[i - 1]

        solver.Add(Ax[i - 1] >= ax)
        solver.Add(Ax[i - 1] >= -ax)

        solver.Add(Ay[i - 1] >= ay)
        solver.Add(Ay[i - 1] >= -ay)

        solver.Add(Aa[i - 1] >= aa)
        solver.Add(Aa[i - 1] >= -aa)

    Jx = [solver.NumVar(-1e6, 1e6, f'Jx_{i}') for i in range(n - 3)]
    Jy = [solver.NumVar(-1e6, 1e6, f'Jy_{i}') for i in range(n - 3)]
    Ja = [solver.NumVar(-1e6, 1e6, f'Ja_{i}') for i in range(n - 3)]

    for i in range(2, n - 1):
        jx = Px[i + 1] - 3 * Px[i] + 3 * Px[i - 1] - Px[i - 2]
        jy = Py[i + 1] - 3 * Py[i] + 3 * Py[i - 1] - Py[i - 2]
        ja = Pa[i + 1] - 3 * Pa[i] + 3 * Pa[i - 1] - Pa[i - 2]

        solver.Add(Jx[i - 2] >= jx)
        solver.Add(Jx[i - 2] >= -jx)

        solver.Add(Jy[i - 2] >= jy)
        solver.Add(Jy[i - 2] >= -jy)

        solver.Add(Ja[i - 2] >= ja)
        solver.Add(Ja[i - 2] >= -ja)

    objective = solver.Objective()
    objective.SetMinimization()

    for v in Vx + Vy + Va:
        objective.SetCoefficient(v, 50)

    for a in Ax + Ay + Aa:
        objective.SetCoefficient(a, 200)

    for j in Jx + Jy + Ja:
        objective.SetCoefficient(j, 3000)

    print(f"Solving with {solver.SolverVersion()}")
    result_status = solver.Solve()

    print(f"Status: {result_status}")
    if result_status != pywraplp.Solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if result_status == pywraplp.Solver.FEASIBLE:
            print("A potentially suboptimal solution was found")
        else:
            print("The solver could not solve the problem.")
            return

    smooth_Cx = [Px[i].solution_value() for i in range(n)]
    smooth_Cy = [Py[i].solution_value() for i in range(n)]
    smooth_Ca = [Pa[i].solution_value() for i in range(n)]

    return smooth_Cx, smooth_Cy, smooth_Ca

def stabilise_video(Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca, video_path="video.mp4", output_path="stabilized.mp4", progress_callback=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    n_frames = len(Cx)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    dx_all = medfilt([smooth_Cx[i] - Cx[i] for i in range(n_frames)], 5)
    dy_all = medfilt([smooth_Cy[i] - Cy[i] for i in range(n_frames)], 5)
    da_all = medfilt([smooth_Ca[i] - Ca[i] for i in range(n_frames)], 5)

    for i in tqdm(range(len(Cx)), desc="Stabilizing"):
        ret, frame = cap.read()
        if not ret:
            break

        if i == 0:
            dx = dy = da = 0
        else:
            dx = dx_all[i]
            dy = dy_all[i]
            da = da_all[i]

            dx = smooth_Cx[i] - Cx[i]
            dy = smooth_Cy[i] - Cy[i]
            da = smooth_Ca[i] - Ca[i]

            # max_shift = 30
            # dx = np.clip(dx, -max_shift, max_shift)
            # dy = np.clip(dy, -max_shift, max_shift)

        cos_a = np.cos(da)
        sin_a = np.sin(da)

        M = np.array([
            [cos_a, -sin_a, dx],
            [sin_a, cos_a, dy]
        ], dtype=np.float32)

        stabilized = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        stabilized = fix_border_to_stable(stabilized, dx, dy, da)
        #stabilized = fix_border(stabilized)
        #print("shift pix", np.max(np.abs(dx)), np.max(np.abs(dy)))

        out.write(stabilized)

        if progress_callback:
            progress_callback(int(100 * (i + 1) / n_frames))

    cap.release()
    out.release()

# if __name__ == "__main__":
#     Cx, Cy, Ca = get_shaky_trajectory()
#     smooth_Cx, smooth_Cy, smooth_Ca = smooth_trajectory(Cx, Cy, Ca)
#     stabilise_video(Cx, Cy, Ca, smooth_Cx, smooth_Cy, smooth_Ca)
    # plt.figure(figsize=(10, 4))
    # plt.plot(Cx, label="x (shaky)")
    # plt.plot(smooth_Cx, label="x (smooth)")
    # plt.plot(Cy, label="y (shaky)")
    # plt.plot(smooth_Cy, label="y (smooth)")
    # plt.plot(np.degrees(Ca), label="angle (shaky)")
    # plt.plot(np.degrees(smooth_Ca), label="angle (smooth)")
    # plt.xlabel("Frame")
    # plt.ylabel("Cumulative motion")
    # plt.title("Camera trajectory")
    # plt.legend()
    # plt.show()
