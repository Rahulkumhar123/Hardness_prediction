import cv2
import numpy as np
import os
import pandas as pd

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, remove_small_objects

PIXEL_TO_UM = 0.0439453


# -----------------------------
# simple preprocessing
# -----------------------------
def preprocess(gray):

    clahe = cv2.createCLAHE(2.5, (8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 2
    )

    binary = thresh > 0
    binary = remove_small_objects(binary, 50)
    

    return binary


# -----------------------------
# ILS calculation
# -----------------------------
def calculate_interlamellar_spacing(gray):

    binary = preprocess(gray)
    skeleton = skeletonize(binary)

    coords = np.column_stack(np.where(skeleton))

    if len(coords) < 200:
        return 0

    tree = cKDTree(coords)

    PCA_RADIUS = 15
    MAX_DISTANCE = 50
    SAMPLE_STEP = 6

    def get_direction(pt):
        idx = tree.query_ball_point(pt, PCA_RADIUS)
        if len(idx) < 10:
            return None

        pts = coords[idx]
        pca = PCA(n_components=2)
        pca.fit(pts)

        vec = pca.components_[0]
        return vec / np.linalg.norm(vec)

    spacings = []

    for y, x in coords[::SAMPLE_STEP]:

        direction = get_direction([y, x])
        if direction is None:
            continue

        dy, dx = direction
        ny, nx = -dx, dy  # perpendicular

        dist_list = []

        for sign in [1, -1]:

            for d in range(2, MAX_DISTANCE):

                yy = int(y + sign * ny * d)
                xx = int(x + sign * nx * d)

                if (yy < 0 or yy >= skeleton.shape[0] or
                xx < 0 or xx >= skeleton.shape[1]):
                    break

                if not skeleton[yy, xx]:
                    continue

                dist_list.append(d)
                break

        if len(dist_list) > 0:
                spacings.append(np.median(dist_list))

    if len(spacings) < 30:
            return 0

    spacings = np.array(spacings)

                # simple filtering
    mean = np.mean(spacings)
    std = np.std(spacings)

    spacings = spacings[(spacings > mean - 2 * std) &
            (spacings < mean + 2 * std)]

    if len(spacings) == 0:
            return 0

    spacing_px = np.median(spacings)
    spacing_um = spacing_px * PIXEL_TO_UM

    return spacing_um


                # -----------------------------
                # batch processing
                # -----------------------------
if __name__ == "__main__":

    folder_path = r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\TileSet"
    results = []

    for file in os.listdir(folder_path):

        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):

            path = os.path.join(folder_path, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("skip:", file)
                continue

            ils = calculate_interlamellar_spacing(img)

            results.append({
                "image": file,
                "ILS_um": ils
            })

            print(file, "->", round(ils, 4), "um")
    df = pd.DataFrame(results)
    df.to_csv("ILS_results.csv", index=False)

print("\ndone! csv saved")