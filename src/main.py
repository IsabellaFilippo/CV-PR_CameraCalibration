import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ── FUNZIONI ────────────────────────────────────────────────

def create_compound_image(rows, cols, limages):
    h, w, c = limages[0].shape
    compound_img = np.empty((rows * h, cols * w, c), dtype="uint8")
    for i in range(rows):
        images_max_index = min((i + 1) * cols, len(limages))
        compound_img_row = np.concatenate(limages[i * cols: images_max_index], axis=1)
        compound_img[i * h: (i + 1) * h, : compound_img_row.shape[1]] = compound_img_row
        if images_max_index == len(limages) - 1:
            break
    return compound_img

def compute_real_coordinates(N, grid_size, square_size):
    """Calcola le coordinate mondo (mm) per N corner della scacchiera."""
    grid_size_cv2 = tuple(reversed(grid_size))
    real_coords = np.zeros((N, 2), dtype=float)
    for idx in range(N):
        u_index, v_index = np.unravel_index(idx, grid_size_cv2)
        real_coords[idx] = [u_index * square_size, v_index * square_size]
    return real_coords

def estimate_homography_dlt(corners, real_coordinates):
    """Stima la matrice di omografia H via DLT (SVD)."""
    A = np.empty((0, 9), dtype=float)
    for i in range(len(corners)):
        u, v = corners[i]
        x, y = real_coordinates[i]
        A = np.vstack((
            A,
            [x, y, 1.0, 0.0, 0.0, 0.0, -u*x, -u*y, -u],
            [0.0, 0.0, 0.0, x, y, 1.0, -v*x, -v*y, -v]
        ))
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H

def project_rect(H, rect_world):
    """Proietta i vertici del rettangolo mondo → pixel tramite H."""
    rect_world_h = np.hstack([rect_world, np.ones((len(rect_world), 1), dtype=float)])
    proj = (H @ rect_world_h.T).T
    return proj[:, :2] / proj[:, 2:3]

def v_ij(H, i, j):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ], dtype=float)

def estimate_K(Hs):
    """Stima la matrice intrinseca K dal metodo di Zhang."""
    V = []
    for H in Hs:
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    if b[-1] < 0:
        b = -b
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return None  # B non positiva definita → caso degenere
    K = np.linalg.inv(L.T)
    K = K / K[2, 2]
    return K
def compute_extrinsics(K, H):
    """Calcola R (ortogonalizzata) e t dagli estrinseci."""
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1  = lam * (K_inv @ h1)
    r2  = lam * (K_inv @ h2)
    r3  = np.cross(r1, r2)
    t   = lam * (K_inv @ h3)
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
        t *= -1
    return R, t

def check_rotation_matrix(R):
    """Verifica ortogonalità e determinante di una matrice di rotazione."""
    RtR = R.T @ R
    I   = np.eye(3)
    print("R^T R =\n", RtR)
    print("\nR^T R - I =\n", RtR - I)
    print("\nDiagonal errors (should be 0):", np.diag(RtR - I))
    off_diag = (RtR - I) - np.diag(np.diag(RtR - I))
    print("Off-diagonal max abs error (should be 0):", np.max(np.abs(off_diag)))
    print("\ndet(R) =", np.linalg.det(R))

def project_points(K, R, t, objp_3d):
    """Proietta punti 3D → pixel."""
    X     = objp_3d.T
    x_cam = (R @ X) + t.reshape(3, 1)
    x_img = K @ x_cam
    return (x_img[:2, :] / x_img[2, :]).T

def make_cylinder(center=(4, 5), radius=2.0, height=5.0, n_points=40):
    """Genera i punti 3D della base e della cima di un cilindro."""
    cx, cy  = center
    angles  = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    base_pts = np.zeros((n_points, 3), dtype=float)
    top_pts  = np.zeros((n_points, 3), dtype=float)
    base_pts[:, 0] = cx + radius * np.cos(angles)
    base_pts[:, 1] = cy + radius * np.sin(angles)
    top_pts[:, 0]  = base_pts[:, 0]
    top_pts[:, 1]  = base_pts[:, 1]
    top_pts[:, 2]  = height
    return base_pts, top_pts

def draw_cylinder(img, K, R, t,
                  center=(4, 5), radius=2.0, height=5.0, n_points=40,
                  base_color=(255, 0, 0), top_color=(0, 0, 255), side_color=(0, 255, 0)):
    """Disegna un cilindro 3D proiettato sull'immagine."""
    base3d, top3d = make_cylinder(center, radius, height, n_points)
    base2d = project_points(K, R, t, base3d)
    top2d  = project_points(K, R, t, top3d)
    out = img.copy()
    N = len(base2d)
    for i in range(N):
        p1 = tuple(np.round(base2d[i]).astype(int))
        p2 = tuple(np.round(base2d[(i + 1) % N]).astype(int))
        q1 = tuple(np.round(top2d[i]).astype(int))
        q2 = tuple(np.round(top2d[(i + 1) % N]).astype(int))
        cv2.line(out, p1, p2, base_color, 2)
        cv2.line(out, q1, q2, top_color,  2)
        cv2.line(out, p1, q1, side_color, 2)
    return out

# ── CONFIGURAZIONE ──────────────────────────────────────────
grid_size   = (8, 11)
square_size = 11          # [mm]
N_corners   = grid_size[0] * grid_size[1]   # 88
folderpath  = "data/images/"
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)

images_path = sorted(
    [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.lower().endswith(".png")],
    key=lambda p: int(p.split('_')[-1].split('.')[0])
)

# ── RETTANGOLO DI RIFERIMENTO ────────────────────────────────
rect_width, rect_height       = 99, 44    # [mm]
bottom_left_x, bottom_left_y = 11, 33    # [mm]

rect_world = np.array([
    [bottom_left_x,               bottom_left_y              ],
    [bottom_left_x + rect_width,  bottom_left_y              ],
    [bottom_left_x + rect_width,  bottom_left_y + rect_height],
    [bottom_left_x,               bottom_left_y + rect_height],
], dtype=float)

# ── COORDINATE E RETTANGOLO IMMAGINE SINGOLA (rgb_0.png) ─────────────
filepath = folderpath + 'rgb_0.png'
image    = cv2.imread(filepath)
image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


return_value, corners = cv2.findChessboardCorners(image, patternSize=grid_size)
corners = corners.reshape((N_corners, 2)).copy()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)


# sovrapponi le coordinate mondo
real_coordinates = compute_real_coordinates(N_corners, grid_size, square_size)
another_copy = image.copy()
for idx, (u_coord, v_coord) in enumerate(corners):
    x_mm, y_mm = real_coordinates[idx]
    cv2.putText(another_copy, text=f"{int(x_mm)};{int(y_mm)}",
                org=(round(u_coord), round(v_coord)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4,
                color=(255, 0, 0), thickness=1)

plt.figure(figsize=(6, 5))
plt.imshow(another_copy)
plt.axis('off')
plt.savefig(os.path.join(figures_dir, "corners_with_coordinates.png"), dpi=150)
# plt.show()

# calcola H per rgb_0 e mostra il rettangolo proiettato
H_single = estimate_homography_dlt(corners, real_coordinates)
rect_img = project_rect(H_single, rect_world)
img_draw = image.copy()
pts = np.round(rect_img).astype(np.int32).reshape(-1, 1, 2)
cv2.polylines(img_draw, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

plt.figure(figsize=(6, 5))
plt.imshow(img_draw)
plt.axis('off')
plt.savefig(os.path.join(figures_dir, "rectangle_single.png"), dpi=150)
#plt.show()

# ── LOOP SU TUTTE LE IMMAGINI: OMOGRAFIE + RETTANGOLO ───────
Hs           = []
good_images  = []
drawn_images = []

for p in images_path:
    im = cv2.imread(p)
    if im is None:
        print(f"Could not read image: {p}")
        continue

    gray  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, patternSize=grid_size)

    if not found:
        print(f"Pattern not found for image {p}")
        continue

    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    corners = corners.reshape((N_corners, 2))

    real_coordinates = compute_real_coordinates(N_corners, grid_size, square_size)
    H = estimate_homography_dlt(corners, real_coordinates)

    Hs.append(H)
    good_images.append(p)

    rect_img = project_rect(H, rect_world)
    img_draw = im.copy()
    pts = np.round(rect_img).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_draw, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
    drawn_images.append(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))

print(f"Computed {len(Hs)} homographies out of {len(images_path)} images.")

compound_image = create_compound_image(rows=5, cols=5, limages=drawn_images)
plt.figure(figsize=(6, 9))
plt.imshow(compound_image)
plt.axis('off')
plt.savefig(os.path.join(figures_dir, "rectangle_compound.png"), dpi=150)
#plt.show()


# ── ZHANG CALIBRATION STEP 1: K ─────────────────────────────
K = estimate_K(Hs)
print("K =\n", K)

# ── ZHANG CALIBRATION STEP 2: ESTRINSECI ────────────────────
Rs, ts = [], []

for H in Hs:
    R, t = compute_extrinsics(K, H)
    Rs.append(R)
    ts.append(t)

Rs_ortho = Rs  # alias: compute_extrinsics restituisce già R ortogonalizzata

print("\nt[0] =", ts[0])
print("R[0] =\n", Rs[0])
check_rotation_matrix(Rs[0])

# ── TASK 2: REPROJECTION ERROR ───────────────────────────────
i_img    = 0
img_path = good_images[i_img]

im   = cv2.imread(img_path)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

found, corners = cv2.findChessboardCorners(gray, patternSize=grid_size)
if not found:
    raise ValueError(f"Pattern not found in selected image: {img_path}")

cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
corners_meas = corners.reshape(N_corners, 2)

real_coordinates = compute_real_coordinates(N_corners, grid_size, square_size)
world_h = np.hstack([real_coordinates, np.ones((N_corners, 1), dtype=float)])

R = Rs[i_img]
t = ts[i_img].reshape(3, 1)
H_pred = K @ np.hstack([R[:, 0:1], R[:, 1:2], t])

proj   = (H_pred @ world_h.T).T
uv_hat = proj[:, :2] / proj[:, 2:3]

du   = uv_hat[:, 0] - corners_meas[:, 0]
dv   = uv_hat[:, 1] - corners_meas[:, 1]
errs = np.sqrt(du**2 + dv**2)

eps_total = float(np.sum(du**2 + dv**2))
mean_err  = float(errs.mean())
max_err   = float(errs.max())

print(f"\nSelected image: {os.path.basename(img_path)}")
print(f"Total reprojection error ε(P) = {eps_total:.4f} px²")
print(f"Mean reprojection error       = {mean_err:.4f} px")
print(f"Max  reprojection error       = {max_err:.4f} px")

img_draw = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
for (u, v), (uh, vh) in zip(corners_meas, uv_hat):
    cv2.circle(img_draw, (int(round(u)),  int(round(v))),  4, (0, 255, 0), -1)  # misurato (verde)
    cv2.circle(img_draw, (int(round(uh)), int(round(vh))), 3, (255, 0, 0), -1)  # riprodotto (rosso)

plt.figure(figsize=(9, 6))
plt.imshow(img_draw)
plt.title(f"Reprojection — mean: {mean_err:.3f} px  max: {max_err:.3f} px")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "reprojection_error.png"), dpi=150)
#plt.show()

# ── TASK 3: SOVRAPPOSIZIONE CILINDRO 3D ─────────────────────
M    = 25
idxs = list(range(min(M, len(good_images))))

print("Using", len(idxs), "images for Task 3.")
print("Example image:", os.path.basename(good_images[idxs[0]]))
print("len(good_images) =", len(good_images))
print("len(Rs_ortho)    =", len(Rs_ortho))
print("len(ts)          =", len(ts))

cylinder_images = []

for i in idxs:
    img = cv2.imread(good_images[i])
    out = draw_cylinder(img, K, Rs_ortho[i], ts[i],
                        center=(40, 50), radius=15.0, height=80.0,
                        base_color=(255, 0, 0), top_color=(0, 0, 255), side_color=(0, 255, 0))
    cv2.imwrite(os.path.join(figures_dir, f"cylinder_{i:02d}.png"), out)
    cylinder_images.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

print("Cilindro disegnato su", len(idxs), "immagini.")

compound_cylinder = create_compound_image(rows=5, cols=5, limages=cylinder_images)
plt.figure(figsize=(12, 12))
plt.imshow(compound_cylinder)
plt.axis('off')
plt.savefig(os.path.join(figures_dir, "cylinder_compound.png"), dpi=150)
#plt.show()

# ── TASK 4: STABILITÀ DEL PRINCIPAL POINT ─────────────────────

rng    = np.random.default_rng(0)
Ns     = list(range(3, len(Hs) + 1)) #servono almeno 3 immagini
trials = 100
n_images      = len(Hs)
std_u0 = []
std_v0 = []

for n in Ns:
    u0_vals = []
    v0_vals = []
    for _ in range(trials):
        idx   = rng.choice(n_images, size=n, replace=False)
        H_sub = [Hs[i] for i in idx]
        K_sub = estimate_K(H_sub)
        if K_sub is None:
            continue
        u0_vals.append(K_sub[0, 2])
        v0_vals.append(K_sub[1, 2])
    if len(u0_vals) < 2:
        std_u0.append(np.nan)
        std_v0.append(np.nan)
    else:
        std_u0.append(np.std(u0_vals, ddof=1))
        std_v0.append(np.std(v0_vals, ddof=1))

plt.figure(figsize=(8, 5))
plt.plot(Ns, std_u0, marker='o', markersize=3, label='std(u0) [cx]')
plt.plot(Ns, std_v0, marker='o', markersize=3, label='std(v0) [cy]')
plt.xlabel("Numero di immagini usate per calibrazione")
plt.ylabel("Deviazione standard (pixel)")
plt.title("Stabilità del principal point vs numero di immagini")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "stability_principal_point.png"), dpi=150)
#plt.show()