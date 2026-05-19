"""
Improved Statue reconstruction.

The 18 statue images form a full 360° orbit but the background changes
dramatically at the transition angles:
  - Images 1-7 + 18: front of statue, brick wall + checker markers
  - Images 8-17:     back of statue, outdoor desert

The two halves share almost no background features, breaking a pure
sequential SfM chain.

Strategy: reconstruct each hemisphere independently, then stack the
outputs (they'll be in different coordinate frames, but both are valid).
"""

import cv2
import numpy as np
import os
import re


_PROJECT = os.path.dirname(os.path.abspath(__file__))
_IMAGES  = os.environ.get("STEM_IMAGES_DIR", os.path.join(_PROJECT, "TestImages"))
BASE     = os.path.join(_IMAGES, "Statue")
OUT      = os.path.join(_PROJECT, "output")

if not os.path.isdir(BASE):
    print(f"ERROR: Statue images folder not found at:\n  {BASE}\n")
    print("Place the TestImages folder inside the project directory, or set")
    print("the STEM_IMAGES_DIR environment variable to its location.")
    print("\nExample (Windows):  set STEM_IMAGES_DIR=C:\\path\\to\\TestImages")
    print("Example (Mac/Linux): export STEM_IMAGES_DIR=/path/to/TestImages")
    raise SystemExit(1)

K_FILE = os.path.join(BASE, "K.txt")


def parse_K(k_file):
    with open(k_file) as f:
        text = f.read()
    nums = re.findall(r'[\d\.]+', text)
    vals = [float(x) for x in nums]
    return np.array([[vals[0], 0, vals[2]],
                     [0, vals[4], vals[5]],
                     [0, 0, 1.0]])


def load_images(folder, prefix, indices, ext="png"):
    imgs = []
    for i in indices:
        p = os.path.join(folder, f"{prefix}{i}.{ext}")
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: could not load {p}")
        imgs.append(img)
    return imgs


def extract_features(images, n=15000):
    sift = cv2.SIFT_create(nfeatures=n)
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


def match_pair(d1, d2, ratio=0.80):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return [], []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(d1, d2, k=2)
    i1, i2 = [], []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                i1.append(m.queryIdx); i2.append(m.trainIdx)
    return i1, i2


def remove_outliers(pts, z_thresh=4.0):
    if len(pts) < 10:
        return pts
    med = np.median(pts[:, :3], axis=0)
    mad = np.median(np.abs(pts[:, :3] - med), axis=0) + 1e-9
    z = np.abs(pts[:, :3] - med) / (mad * 1.4826)
    return pts[np.all(z < z_thresh, axis=1)]


def sfm_chain(images, K, kps_list, descs_list, label="", window=4, max_depth=150):
    n = len(images)
    R_g = [None] * n
    t_g = [None] * n
    R_g[0] = np.eye(3)
    t_g[0] = np.zeros((3, 1))
    init = [False] * n
    init[0] = True

    # ── Phase 1: pose chain (consecutive only) ────────────────────────────
    for i in range(n - 1):
        j = i + 1
        if images[i] is None or images[j] is None:
            if init[i]: R_g[j]=R_g[i].copy(); t_g[j]=t_g[i].copy(); init[j]=True
            continue
        i1, i2 = match_pair(descs_list[i], descs_list[j])
        if len(i1) < 8:
            if init[i]: R_g[j]=R_g[i].copy(); t_g[j]=t_g[i].copy(); init[j]=True
            continue
        p1 = np.float32([kps_list[i][k].pt for k in i1])
        p2 = np.float32([kps_list[j][k].pt for k in i2])
        E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or mask is None:
            if init[i]: R_g[j]=R_g[i].copy(); t_g[j]=t_g[i].copy(); init[j]=True
            continue
        m = mask.ravel().astype(bool)
        p1i, p2i = p1[m], p2[m]
        if len(p1i) < 5:
            if init[i]: R_g[j]=R_g[i].copy(); t_g[j]=t_g[i].copy(); init[j]=True
            continue
        _, R_rel, t_rel, _ = cv2.recoverPose(E, p1i, p2i, K)
        if init[i]:
            R_g[j] = R_rel @ R_g[i]
            t_g[j] = R_rel @ t_g[i] + t_rel
        else:
            R_g[j] = R_rel.copy(); t_g[j] = t_rel.copy()
        init[j] = True
        print(f"  [{label}] Pose {i+1}-{j+1}: {len(p1i)} inliers")

    # ── Phase 2: triangulate within window ────────────────────────────────
    all_pts = []
    for i in range(n):
        if not init[i] or images[i] is None: continue
        for j in range(i+1, min(i+1+window, n)):
            if not init[j] or images[j] is None: continue
            i1, i2 = match_pair(descs_list[i], descs_list[j])
            if len(i1) < 8: continue
            p1 = np.float32([kps_list[i][k].pt for k in i1])
            p2 = np.float32([kps_list[j][k].pt for k in i2])
            E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
            if E is None or mask is None: continue
            m = mask.ravel().astype(bool)
            p1i, p2i = p1[m], p2[m]
            if len(p1i) < 5: continue
            _, _, _, pm = cv2.recoverPose(E, p1i, p2i, K)
            pm = pm.ravel().astype(bool)
            p1t, p2t = p1i[pm], p2i[pm]
            if len(p1t) < 4: continue
            P1 = K @ np.hstack([R_g[i], t_g[i]])
            P2 = K @ np.hstack([R_g[j], t_g[j]])
            h4 = cv2.triangulatePoints(P1, P2, p1t.T, p2t.T)
            w = h4[3]
            ok = np.abs(w) > 1e-10
            pts3 = (h4[:3, ok] / w[ok]).T
            p1ok = p1t[ok]
            added = 0
            for k, pt in enumerate(pts3):
                zi = (R_g[i] @ pt + t_g[i].ravel())[2]
                zj = (R_g[j] @ pt + t_g[j].ravel())[2]
                if zi <= 0 or zj <= 0 or zi > max_depth or zj > max_depth:
                    continue
                cx, cy = p1ok[k]
                H, W = images[i].shape[:2]
                ri = int(np.clip(cy, 0, H-1))
                ci_ = int(np.clip(cx, 0, W-1))
                bgr = images[i][ri, ci_]
                all_pts.append([pt[0], pt[1], pt[2],
                                 int(bgr[2]), int(bgr[1]), int(bgr[0])])
                added += 1
            print(f"  [{label}] Tri {i+1}-{j+1}: {added} pts")

    pts = np.array(all_pts) if all_pts else np.zeros((0, 6))
    if len(pts) > 10:
        pts = remove_outliers(pts)
    return pts


def save_txt(points, path):
    with open(path, 'w') as f:
        f.write("X Y Z R G B\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(p[3])} {int(p[4])} {int(p[5])}\n")
    print(f"  Saved {len(points)} pts → {path}")


def save_ply(points, path):
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(p[3])} {int(p[4])} {int(p[5])}\n")
    print(f"  PLY → {path}")


def save_viz(points, path, title):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    if len(points) == 0: return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, len(points) // 20000)
    ax.scatter(points[::step,0], points[::step,1], points[::step,2],
               c=np.clip(points[::step,3:6]/255., 0, 1), s=0.5, linewidths=0)
    ax.set_title(title); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Viz → {path}")


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    K = parse_K(K_FILE)
    print(f"K =\n{K}\n")

    # Front hemisphere: images 1-7 + 18 (brick wall, checker markers)
    # Use ordering: 18, 1, 2, 3, 4, 5, 6, 7 so chain is coherent
    front_idx = [18, 1, 2, 3, 4, 5, 6, 7]
    print("=== Front hemisphere (brick wall) ===")
    front_imgs = load_images(BASE, "statue", front_idx, "png")
    front_kps, front_descs = extract_features(front_imgs)
    front_pts = sfm_chain(front_imgs, K, front_kps, front_descs, "front")
    print(f"  Front points: {len(front_pts)}")

    # Back hemisphere: images 8-17 (desert)
    back_idx = list(range(8, 18))
    print("\n=== Back hemisphere (desert) ===")
    back_imgs = load_images(BASE, "statue", back_idx, "png")
    back_kps, back_descs = extract_features(back_imgs)
    back_pts = sfm_chain(back_imgs, K, back_kps, back_descs, "back")
    print(f"  Back points: {len(back_pts)}")

    # Save each half separately (different coordinate frames)
    save_txt(front_pts, os.path.join(OUT, "statue_front_points.txt"))
    save_ply(front_pts, os.path.join(OUT, "statue_front_points.ply"))
    save_viz(front_pts, os.path.join(OUT, "statue_front_visualization.png"),
             "Statue (front — brick wall)")

    save_txt(back_pts,  os.path.join(OUT, "statue_back_points.txt"))
    save_ply(back_pts,  os.path.join(OUT, "statue_back_points.ply"))
    save_viz(back_pts,  os.path.join(OUT, "statue_back_visualization.png"),
             "Statue (back — desert)")

    # Also save combined (two separate coordinate frames, noted in filename)
    combined = np.vstack([front_pts, back_pts]) if len(front_pts) and len(back_pts) else (
        front_pts if len(front_pts) else back_pts)
    save_txt(combined, os.path.join(OUT, "statue_combined_points.txt"))
    save_ply(combined, os.path.join(OUT, "statue_combined_points.ply"))
    print(f"\nTotal statue points: {len(combined)}")
