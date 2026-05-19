"""
3D Point Cloud Reconstruction for Statue and Fountain datasets.
Camera intrinsics (K) are known but extrinsics are unknown.
Phase 1: chain camera poses from consecutive pairs via essential matrix.
Phase 2: triangulate using pairs within a sliding window.
"""

import cv2
import numpy as np
import os
import re


def parse_K(k_file):
    with open(k_file) as f:
        text = f.read()
    nums = re.findall(r'[\d\.]+', text)
    vals = [float(x) for x in nums]
    return np.array([
        [vals[0], 0,       vals[2]],
        [0,       vals[4], vals[5]],
        [0,       0,       1.0   ]
    ])


def extract_features(images, n_features=15000):
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps_list, descs_list = [], []
    for img in images:
        if img is None:
            kps_list.append([]); descs_list.append(None)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)
        kps_list.append(kps)
        descs_list.append(descs)
    return kps_list, descs_list


def match_pair(descs1, descs2, ratio=0.8):
    if descs1 is None or descs2 is None or len(descs1) < 2 or len(descs2) < 2:
        return [], []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(descs1, descs2, k=2)
    idx1, idx2 = [], []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
    return idx1, idx2


def remove_outliers(points, z_thresh=4.0):
    if len(points) < 10:
        return points
    coords = points[:, :3]
    med = np.median(coords, axis=0)
    mad = np.median(np.abs(coords - med), axis=0) + 1e-9
    z = np.abs(coords - med) / (mad * 1.4826)
    mask = np.all(z < z_thresh, axis=1)
    return points[mask]


def incremental_sfm(images, K, kps_list, descs_list, window=3):
    """
    Phase 1: build camera poses by chaining consecutive pairs.
    Phase 2: triangulate points using all pairs within `window`.
    """
    n = len(images)
    all_points = []

    R_global = [None] * n
    t_global = [None] * n
    R_global[0] = np.eye(3)
    t_global[0] = np.zeros((3, 1))
    initialized = [False] * n
    initialized[0] = True

    # ── Phase 1: build pose chain ──────────────────────────────────────────
    for i in range(n - 1):
        j = i + 1
        if images[i] is None or images[j] is None:
            if initialized[i]:
                R_global[j] = R_global[i].copy()
                t_global[j] = t_global[i].copy()
                initialized[j] = True
            continue

        idx1, idx2 = match_pair(descs_list[i], descs_list[j])
        if len(idx1) < 8:
            print(f"  Pose {i+1}-{j+1}: too few matches ({len(idx1)}), using prev pose")
            if initialized[i]:
                R_global[j] = R_global[i].copy()
                t_global[j] = t_global[i].copy()
                initialized[j] = True
            continue

        pts1 = np.float32([kps_list[i][k].pt for k in idx1])
        pts2 = np.float32([kps_list[j][k].pt for k in idx2])

        E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            if initialized[i]:
                R_global[j] = R_global[i].copy()
                t_global[j] = t_global[i].copy()
                initialized[j] = True
            continue

        inliers = pts1[mask.ravel().astype(bool)]
        inliers2 = pts2[mask.ravel().astype(bool)]
        if len(inliers) < 5:
            if initialized[i]:
                R_global[j] = R_global[i].copy()
                t_global[j] = t_global[i].copy()
                initialized[j] = True
            continue

        _, R_rel, t_rel, _ = cv2.recoverPose(E, inliers, inliers2, K)

        if initialized[i]:
            R_global[j] = R_rel @ R_global[i]
            t_global[j] = R_rel @ t_global[i] + t_rel
        else:
            R_global[j] = R_rel.copy()
            t_global[j] = t_rel.copy()
        initialized[j] = True
        print(f"  Pose {i+1}-{j+1}: {len(inliers)} inliers ✓")

    # ── Phase 2: triangulate within sliding window ─────────────────────────
    for i in range(n):
        if not initialized[i] or images[i] is None:
            continue
        for j in range(i + 1, min(i + 1 + window, n)):
            if not initialized[j] or images[j] is None:
                continue

            idx1, idx2 = match_pair(descs_list[i], descs_list[j])
            if len(idx1) < 8:
                continue

            pts1 = np.float32([kps_list[i][k].pt for k in idx1])
            pts2 = np.float32([kps_list[j][k].pt for k in idx2])

            E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                            method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None or mask is None:
                continue

            mask_bool = mask.ravel().astype(bool)
            pts1_in = pts1[mask_bool]
            pts2_in = pts2[mask_bool]
            if len(pts1_in) < 5:
                continue

            _, _, _, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
            pm = pose_mask.ravel().astype(bool)
            pts1_tri = pts1_in[pm]
            pts2_tri = pts2_in[pm]
            if len(pts1_tri) < 4:
                continue

            P1 = K @ np.hstack([R_global[i], t_global[i]])
            P2 = K @ np.hstack([R_global[j], t_global[j]])

            pts4d = cv2.triangulatePoints(P1, P2, pts1_tri.T, pts2_tri.T)
            w = pts4d[3]
            valid_w = np.abs(w) > 1e-10
            pts3d = (pts4d[:3, valid_w] / w[valid_w]).T
            pts1_valid = pts1_tri[valid_w]

            added = 0
            for k, pt in enumerate(pts3d):
                z_i = (R_global[i] @ pt + t_global[i].ravel())[2]
                z_j = (R_global[j] @ pt + t_global[j].ravel())[2]
                if z_i <= 0 or z_j <= 0:
                    continue
                if z_i > 200 or z_j > 200:
                    continue

                col_f, row_f = pts1_valid[k]
                h, w_ = images[i].shape[:2]
                ri = int(np.clip(row_f, 0, h - 1))
                ci = int(np.clip(col_f, 0, w_ - 1))
                bgr = images[i][ri, ci]
                all_points.append([pt[0], pt[1], pt[2],
                                    int(bgr[2]), int(bgr[1]), int(bgr[0])])
                added += 1
            print(f"  Tri {i+1}-{j+1}: {added} points")

    print(f"  Raw 3D points: {len(all_points)}")
    pts = np.array(all_points) if all_points else np.zeros((0, 6))
    if len(pts) > 10:
        pts = remove_outliers(pts, z_thresh=4.0)
        print(f"  After outlier removal: {len(pts)}")
    return pts


def save_txt(points, out_path):
    with open(out_path, 'w') as f:
        f.write("X Y Z R G B\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(p[3])} {int(p[4])} {int(p[5])}\n")
    print(f"  Saved {len(points)} points → {out_path}")


def save_ply(points, out_path):
    n = len(points)
    with open(out_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(p[3])} {int(p[4])} {int(p[5])}\n")
    print(f"  Saved PLY → {out_path}")


def save_matplotlib(points, out_path, title="Point Cloud"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    if len(points) == 0:
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, len(points) // 20000)
    xs, ys, zs = points[::step, 0], points[::step, 1], points[::step, 2]
    colors = np.clip(points[::step, 3:6] / 255.0, 0, 1)
    ax.scatter(xs, ys, zs, c=colors, s=0.5, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved visualization → {out_path}")


if __name__ == '__main__':
    _project = os.path.dirname(os.path.abspath(__file__))
    base = os.environ.get("STEM_IMAGES_DIR",
                          os.path.join(_project, "TestImages"))
    out_base = os.path.join(_project, "output")

    if not os.path.isdir(base):
        print(f"ERROR: TestImages folder not found at:\n  {base}\n")
        print("Place the TestImages folder inside the project directory, or set")
        print("the STEM_IMAGES_DIR environment variable to its location.")
        print("\nExample (Windows):  set STEM_IMAGES_DIR=C:\\path\\to\\TestImages")
        print("Example (Mac/Linux): export STEM_IMAGES_DIR=/path/to/TestImages")
        raise SystemExit(1)

    os.makedirs(out_base, exist_ok=True)

    datasets = [
        {'name': 'Statue',   'folder': os.path.join(base, 'Statue'),   'prefix': 'statue',   'ext': 'png'},
        {'name': 'Fountain', 'folder': os.path.join(base, 'Fountain'), 'prefix': 'fountain', 'ext': 'jpg'},
    ]

    for ds in datasets:
        print(f"\n=== Processing {ds['name']} ===")
        folder = ds['folder']
        K = parse_K(os.path.join(folder, 'K.txt'))
        print(f"  K =\n{K}")

        files = sorted(
            [f for f in os.listdir(folder)
             if f.startswith(ds['prefix']) and f.endswith(ds['ext'])],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        images = [cv2.imread(os.path.join(folder, f)) for f in files]
        print(f"  Loaded {sum(1 for x in images if x is not None)}/{len(files)} images")

        print("  Extracting features...")
        kps_list, descs_list = extract_features(images)

        print("  Running incremental SfM...")
        points = incremental_sfm(images, K, kps_list, descs_list, window=3)
        print(f"  Final 3D points: {len(points)}")

        name = ds['name'].lower()
        save_txt(points, os.path.join(out_base, f"{name}_points.txt"))
        save_ply(points, os.path.join(out_base, f"{name}_points.ply"))
        save_matplotlib(points, os.path.join(out_base, f"{name}_visualization.png"), title=ds['name'])
