# Camera Calibration — Zhang's Method
Project 1 – Computer Vision and Pattern Recognition – Academic Year 2025/2026

## Description
Implementation of Zhang's camera calibration from scratch using 81 chessboard images. Homographies are estimated via DLT, K is recovered through Cholesky factorization, and extrinsic parameters (R, t) are computed analytically. Validated through reprojection error and 3D cylinder superimposition.

## Requirements
- Python 3.12
- numpy
- opencv-python-headless
- matplotlib

## Option 1 — Run on GitHub Codespaces (recommended)

1. Open the repository on GitHub
2. Click the green **Code** button
3. Select the **Codespaces** tab
4. Click **Create codespace on main**
5. Wait for the environment to load, then in the terminal run:
pip install -r requirements.txt
python src/main.py

## Option 2 — Run locally

1. Clone the repository:
git clone <your-repo-url>
cd CV-PR_CameraCalibration

2. Install dependencies:
pip install -r requirements.txt

3. Make sure the calibration images are in the following folder:
data/images/

4. Run:
python src/main.py

## Output
All figures are saved in the `figures/` directory:
- `corners_with_coordinates.png` — checkerboard corners with world coordinates
- `rectangle_single.png` — projected rectangle on a single image
- `rectangle_compound.png` — projected rectangle on all images (5x5 mosaic)
- `reprojection_error.png` — measured vs reprojected corners
- `cylinder_compound.png` — 3D cylinder superimposed on 25 images (5x5 mosaic)
- `stability_principal_point.png` — standard deviation of principal point vs number of images
