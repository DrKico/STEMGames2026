"""
3D Point Cloud Reconstruction for Box and Entrance datasets.
Camera positions and orientations are known. Uses SIFT feature matching
and ray triangulation to recover 3D points.
"""

import cv2
import numpy as np
import os
import re
import struct
from itertools import combinations


# ─── Camera model from problem statement ─────────────────────────────────────

def get_ray_direction(pixel_row, pixel_col, res_x, res_y,
                      cam_forward, cam_right, cam_up):
    """Returns the unit direction vector for a pixel ray."""
    coeff_right = (2.0 * (pixel_col - res_x / 2.0 + 0.5) / res_x)
    coeff_up    = (-2.0 * (pixel_row - res_y / 2.0 + 0.5) / res_y)
    coeff_up   *= res_y / res_x
    direction = cam_forward + coeff_right * cam_right + coeff_up * cam_up
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-9 else direction


def triangulate_rays(o1, d1, o2, d2, max_depth=5000.0):
    """
    Find the closest point between two rays: o1+t*d1 and o2+s*d2.
    Returns (midpoint, distance) or (None, inf) if invalid.
    """
    d1d2 = np.dot(d1, d2)
    denom = 1.0 - d1d2 ** 2
    if abs(denom) < 1e-10:
        return None, float('inf')  # parallel rays

    o12 = o2 - o1
    t = (np.dot(o12, d1) - d1d2 * np.dot(o12, d2)) / denom
    s = (d1d2 * np.dot(o12, d1) - np.dot(o12, d2)) / denom

    # Both intersection parameters must be positive (point in front of cameras)
    if t <= 0 or s <= 0:
        return None, float('inf')
    # Discard points unreasonably far from cameras
    if t > max_depth or s > max_depth:
        return None, float('inf')

    p1 = o1 + t * d1
    p2 = o2 + s * d2
    distance = np.linalg.norm(p2 - p1)
    midpoint = (p1 + p2) / 2.0
    return midpoint, distance


# ─── Camera data parser ────────────────────────────────────────────────────────

def parse_camera_file(filepath):
    """Parse the camera txt file, returns list of camera dicts."""
    with open(filepath) as f:
        text = f.read()

    cameras = []
    # Split by numbered camera sections
    blocks = re.split(r'\n\s*\d+\)\s*\n', text)
    # Find all camera blocks
    pattern = re.compile(
        r'CamPosition:\s*X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)\s*\n'
        r'\s*CamForward:\s*X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)\s*\n'
        r'\s*CamRight[:\s]+X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)\s*\n'
        r'\s*CamUp[:\s]+X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)',
        re.MULTILINE
    )
    for m in pattern.finditer(text):
        vals = [float(x) for x in m.groups()]
        cameras.append({
            'position': np.array(vals[0:3]),
            'forward':  np.array(vals[3:6]),
            'right':    np.array(vals[6:9]),
            'up':       np.array(vals[9:12]),
        })
    return cameras


# ─── Feature matching ──────────────────────────────────────────────────────────

def extract_keypoints(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kps, descs = sift.detectAndCompute(gray, None)
    return kps, descs


def match_images(kps1, descs1, kps2, descs2, ratio=0.8):
    if descs1 is None or descs2 is None or len(kps1) < 2 or len(kps2) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(descs1, descs2, k=2)
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


# ─── Main reconstruction ───────────────────────────────────────────────────────

def remove_outliers(points, z_thresh=3.0):
    """Remove points outside z_thresh standard deviations from the median per axis."""
    if len(points) < 10:
        return points
    coords = points[:, :3]
    med = np.median(coords, axis=0)
    mad = np.median(np.abs(coords - med), axis=0) + 1e-9
    # Robust z-score
    z = np.abs(coords - med) / (mad * 1.4826)
    mask = np.all(z < z_thresh, axis=1)
    return points[mask]


def reconstruct(folder, input_file, image_prefix, image_ext='png',
                res_x=1920, res_y=1080, max_ray_dist=50.0):
    print(f"\n=== Processing {folder} ===")
    cameras = parse_camera_file(input_file)
    print(f"  Loaded {len(cameras)} cameras")

    # Load images
    images = []
    for i in range(1, len(cameras) + 1):
        path = os.path.join(folder, f"{image_prefix}{i}.{image_ext}")
        img = cv2.imread(path)
        if img is None:
            print(f"  WARNING: could not load {path}")
        images.append(img)
    print(f"  Loaded {sum(1 for x in images if x is not None)} images")

    sift = cv2.SIFT_create(nfeatures=20000)

    # Extract features from all images
    print("  Extracting SIFT features...")
    all_kps = []
    all_descs = []
    for img in images:
        if img is not None:
            kps, descs = extract_keypoints(img, sift)
        else:
            kps, descs = [], None
        all_kps.append(kps)
        all_descs.append(descs)

    points_3d = []   # list of [X, Y, Z, R, G, B]

    # Match all pairs of images
    n = len(cameras)
    pairs = list(combinations(range(n), 2))
    print(f"  Matching {len(pairs)} image pairs...")

    for i, j in pairs:
        if images[i] is None or images[j] is None:
            continue

        matches = match_images(all_kps[i], all_descs[i],
                               all_kps[j], all_descs[j])
        if len(matches) < 8:
            continue

        cam_i = cameras[i]
        cam_j = cameras[j]

        for m in matches:
            # Pixel coords in image i (query)
            pt_i = all_kps[i][m.queryIdx].pt  # (col, row)
            col_i, row_i = pt_i

            # Pixel coords in image j (train)
            pt_j = all_kps[j][m.trainIdx].pt
            col_j, row_j = pt_j

            # Ray from camera i through pixel (row_i, col_i)
            d_i = get_ray_direction(row_i, col_i, res_x, res_y,
                                    cam_i['forward'], cam_i['right'], cam_i['up'])
            # Ray from camera j
            d_j = get_ray_direction(row_j, col_j, res_x, res_y,
                                    cam_j['forward'], cam_j['right'], cam_j['up'])

            pt3d, dist = triangulate_rays(cam_i['position'], d_i,
                                          cam_j['position'], d_j)

            if pt3d is None or dist > max_ray_dist:
                continue

            # Color from image i (BGR → RGB)
            r_idx = int(np.clip(row_i, 0, res_y - 1))
            c_idx = int(np.clip(col_i, 0, res_x - 1))
            bgr = images[i][r_idx, c_idx]
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

            points_3d.append([pt3d[0], pt3d[1], pt3d[2], rgb[0], rgb[1], rgb[2]])

    print(f"  Raw 3D points: {len(points_3d)}")
    pts = np.array(points_3d) if points_3d else np.zeros((0, 6))
    if len(pts) > 0:
        pts = remove_outliers(pts, z_thresh=4.0)
        print(f"  After outlier removal: {len(pts)}")
    return pts


# ─── Output writers ───────────────────────────────────────────────────────────

def save_txt(points, out_path):
    """Save XYZ (and optionally RGB) as space-separated text."""
    with open(out_path, 'w') as f:
        f.write("X Y Z R G B\n")
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[3])} {int(p[4])} {int(p[5])}\n")
    print(f"  Saved {len(points)} points → {out_path}")


def save_ply(points, out_path):
    """Save as binary PLY for quick viewing in MeshLab / CloudCompare."""
    n = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(out_path, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[3])} {int(p[4])} {int(p[5])}\n")
    print(f"  Saved PLY → {out_path}")


def save_matplotlib(points, out_path, title="Point Cloud"):
    """Save a 3D scatter plot PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    if len(points) == 0:
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    colors = points[:, 3:6] / 255.0
    # Subsample for speed
    step = max(1, len(points) // 20000)
    ax.scatter(xs[::step], ys[::step], zs[::step],
               c=colors[::step], s=0.5, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved visualization → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _project = os.path.dirname(os.path.abspath(__file__))
    base = os.environ.get("STEM_IMAGES_DIR",
                          os.path.join(_project, "TestImages"))
    out_base = os.path.join(_project, "output")
    os.makedirs(out_base, exist_ok=True)

    datasets = [
        {
            'name': 'Box',
            'folder': os.path.join(base, 'Box'),
            'input':  os.path.join(base, 'Box', 'boxInput.txt'),
            'prefix': 'box',
            'ext':    'png',
        },
        {
            'name': 'Entrance',
            'folder': os.path.join(base, 'Entrance'),
            'input':  os.path.join(base, 'Entrance', 'entranceInput.txt'),
            'prefix': 'entrance',
            'ext':    'png',
        },
    ]

    for ds in datasets:
        points = reconstruct(
            folder=ds['folder'],
            input_file=ds['input'],
            image_prefix=ds['prefix'],
            image_ext=ds['ext'],
        )
        name = ds['name'].lower()
        save_txt(points, os.path.join(out_base, f"{name}_points.txt"))
        save_ply(points, os.path.join(out_base, f"{name}_points.ply"))
        save_matplotlib(points, os.path.join(out_base, f"{name}_visualization.png"),
                        title=ds['name'])
