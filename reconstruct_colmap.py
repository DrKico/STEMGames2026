"""
High-quality 3D reconstruction using pycolmap (COLMAP backend).
Handles all four datasets:
  - Box, Entrance: known cameras (90° FoV, 1920x1080)
  - Statue:        known K, unknown poses (90° FoV, 1920x1080)
  - Fountain:      known K, unknown poses (~84° FoV, 3072x2048)
"""

import pycolmap
import numpy as np
import os
import re
import shutil
from pathlib import Path


OUT = Path("/Users/kmatic/IdeaProjects/STEMgames 2026/output")
BASE = Path("/Users/kmatic/Downloads/StemGames2026_ProjectTask/TestImages")
OUT.mkdir(exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_camera_file(filepath):
    with open(filepath) as f:
        text = f.read()
    pattern = re.compile(
        r'CamPosition:\s*X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)\s*\n'
        r'\s*CamForward:\s*X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)\s*\n'
        r'\s*CamRight[:\s]+X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)\s*\n'
        r'\s*CamUp[:\s]+X=([\d\.\-]+)\s+Y=([\d\.\-]+)\s+Z=([\d\.\-]+)',
        re.MULTILINE
    )
    cameras = []
    for m in pattern.finditer(text):
        v = [float(x) for x in m.groups()]
        cameras.append({
            'position': np.array(v[0:3]),
            'forward':  np.array(v[3:6]),
            'right':    np.array(v[6:9]),
            'up':       np.array(v[9:12]),
        })
    return cameras


def cam_vectors_to_rotation(forward, right, up):
    """
    Convert game-engine camera basis vectors to a rotation matrix R such that
    the camera-to-world transform is [right | up_corrected | forward] but
    COLMAP uses the OpenCV convention: X right, Y down, Z forward.
    """
    # Game engine: right=+X, up=+Y, forward=+Z
    # OpenCV: right=+X, down=+Y, forward=+Z
    # Our forward vector points toward the scene (into screen)
    # Our up vector points up in world space
    # Map: col_x = right, col_y = -up (down), col_z = forward
    R_world_to_cam = np.stack([right, -up, forward], axis=0)  # (3,3)
    return R_world_to_cam  # This is R in P = K[R|t]


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
    if len(points) == 0:
        print(f"  No points for {title}"); return
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, len(points) // 30000)
    c = np.clip(points[::step, 3:6] / 255., 0, 1)
    ax.scatter(points[::step,0], points[::step,1], points[::step,2],
               c=c, s=0.5, linewidths=0)
    ax.set_title(f"{title} ({len(points)} points)")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Viz → {path}")


def extract_colmap_points(reconstruction):
    """Pull XYZ+RGB out of a pycolmap Reconstruction."""
    pts = []
    for pid, p3d in reconstruction.points3D.items():
        xyz = p3d.xyz
        rgb = p3d.color  # uint8 array [R, G, B]
        pts.append([xyz[0], xyz[1], xyz[2], int(rgb[0]), int(rgb[1]), int(rgb[2])])
    return np.array(pts) if pts else np.zeros((0, 6))


# ─── Generic reconstruction (unknown cameras, known K) ────────────────────────

def reconstruct_unknown_cam(name, image_folder, camera_model, camera_params,
                             match_method='exhaustive'):
    print(f"\n{'='*60}")
    print(f"  {name}  (unknown cameras, using COLMAP SfM)")
    print(f"{'='*60}")

    work_dir = OUT / f"colmap_{name.lower()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir()

    # Feature extraction with known intrinsics
    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.max_num_features = 20000
    feat_opts = pycolmap.FeatureExtractionOptions()
    feat_opts.sift = sift_opts

    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model
    reader_opts.camera_params = camera_params

    print("  Extracting features...")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_folder),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=feat_opts,
    )

    # Feature matching
    print(f"  Matching ({match_method})...")
    if match_method == 'exhaustive':
        pycolmap.match_exhaustive(str(db_path))
    else:
        seq_opts = pycolmap.SequentialPairingOptions()
        seq_opts.loop_detection = True
        pycolmap.match_sequential(str(db_path),
                                   pairing_options=seq_opts)

    # Incremental mapping
    print("  Running incremental mapping (COLMAP SfM)...")
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_folder),
        output_path=str(sparse_dir),
        options=mapper_opts,
    )

    if not reconstructions:
        print("  No reconstruction produced.")
        return np.zeros((0, 6))

    # Use the largest reconstruction
    best = max(reconstructions.values(), key=lambda r: len(r.points3D))
    print(f"  Registered {len(best.images)} / {best.summary()}")
    pts = extract_colmap_points(best)
    print(f"  3D points: {len(pts)}")
    return pts


# ─── Known-camera reconstruction ─────────────────────────────────────────────

def reconstruct_known_cam(name, image_folder, cam_data_file, image_prefix, ext,
                           res_x=1920, res_y=1080):
    """
    For Box and Entrance: camera poses are known.
    We still use pycolmap for feature extraction + matching + triangulation,
    but we fix the camera poses from the provided data.
    """
    print(f"\n{'='*60}")
    print(f"  {name}  (known cameras, COLMAP triangulation)")
    print(f"{'='*60}")

    cameras = parse_camera_file(cam_data_file)
    n = len(cameras)

    work_dir = OUT / f"colmap_{name.lower()}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True)

    # For 90° horizontal FoV: fx = res_x/2
    fx = res_x / 2.0
    fy = res_x / 2.0  # square pixels
    cx = res_x / 2.0
    cy = res_y / 2.0
    cam_params = f"{fx},{fy},{cx},{cy}"

    # Extract features
    feat_opts = pycolmap.SiftExtractionOptions()
    feat_opts.max_num_features = 20000
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "PINHOLE"
    reader_opts.camera_params = cam_params
    reader_opts.single_camera = True

    print("  Extracting features...")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_folder),
        image_reader_options=reader_opts,
        sift_options=feat_opts,
    )

    # Match exhaustively
    print("  Matching exhaustively...")
    pycolmap.match_exhaustive(str(db_path))

    # Build reconstruction with known poses
    print("  Building reconstruction with known poses...")
    db = pycolmap.Database(str(db_path))
    reconstruction = pycolmap.Reconstruction()

    # Add camera
    cam = pycolmap.Camera()
    cam.camera_id = 1
    cam.model = pycolmap.CameraModelId.PINHOLE
    cam.width = res_x
    cam.height = res_y
    cam.params = [fx, fy, cx, cy]
    reconstruction.add_camera(cam)

    # Add images with known poses
    db_images = {img.name: img for img in db.read_all_images()}
    for idx, cam_info in enumerate(cameras):
        img_name = f"{image_prefix}{idx+1}.{ext}"
        if img_name not in db_images:
            print(f"  WARNING: {img_name} not in database")
            continue

        db_img = db_images[img_name]
        R = cam_vectors_to_rotation(cam_info['forward'],
                                     cam_info['right'],
                                     cam_info['up'])
        t = -R @ cam_info['position']

        image = pycolmap.Image()
        image.image_id = db_img.image_id
        image.name = img_name
        image.camera_id = 1

        # Set pose: cam_from_world
        quat = pycolmap.Rotation3d(R)
        image.cam_from_world = pycolmap.Rigid3d(quat, t)

        # Load keypoints from database
        kps_db = db.read_keypoints(db_img.image_id)
        pts2d = []
        for kp in kps_db:
            p = pycolmap.Point2D()
            p.xy = np.array([kp[0], kp[1]])
            pts2d.append(p)
        image.points2D = pycolmap.Point2DList(pts2d)
        image.registered = True
        reconstruction.add_image(image)

    db.close()

    # Triangulate
    print("  Triangulating points...")
    tri_opts = pycolmap.IncrementalTriangulatorOptions()
    triangulator = pycolmap.IncrementalTriangulator(
        pycolmap.DatabaseCache.create(
            pycolmap.Database(str(db_path)),
            pycolmap.DatabaseCacheOptions(),
        ),
        reconstruction,
    )
    for image in reconstruction.images.values():
        triangulator.triangulate_image(tri_opts, image.image_id)

    pts = extract_colmap_points(reconstruction)
    print(f"  3D points: {len(pts)}")
    return pts


# ─── Run all datasets ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Box (known cameras, 90° FoV, 1920x1080) ───────────────────────────────
    box_pts = reconstruct_unknown_cam(
        name="Box",
        image_folder=BASE / "Box",
        camera_model="PINHOLE",
        camera_params="960,960,960,540",
        match_method='exhaustive',
    )
    save_txt(box_pts, OUT / "box_colmap_points.txt")
    save_ply(box_pts, OUT / "box_colmap_points.ply")
    save_viz(box_pts, OUT / "box_colmap_visualization.png", "Box (COLMAP)")

    # ── Entrance (known cameras, 90° FoV, 1920x1080) ──────────────────────────
    entrance_pts = reconstruct_unknown_cam(
        name="Entrance",
        image_folder=BASE / "Entrance",
        camera_model="PINHOLE",
        camera_params="960,960,960,540",
        match_method='exhaustive',
    )
    save_txt(entrance_pts, OUT / "entrance_colmap_points.txt")
    save_ply(entrance_pts, OUT / "entrance_colmap_points.ply")
    save_viz(entrance_pts, OUT / "entrance_colmap_visualization.png", "Entrance (COLMAP)")

    # Keep the original variable names for the block below

    # ── Statue (unknown cameras, known K=960/960 PINHOLE 90° FoV) ────────────
    statue_pts = reconstruct_unknown_cam(
        name="Statue",
        image_folder=BASE / "Statue",
        camera_model="PINHOLE",
        camera_params="960,960,960,540",
        match_method='exhaustive',
    )
    save_txt(statue_pts, OUT / "statue_colmap_points.txt")
    save_ply(statue_pts, OUT / "statue_colmap_points.ply")
    save_viz(statue_pts, OUT / "statue_colmap_visualization.png", "Statue (COLMAP)")

    # ── Fountain (unknown cameras, K from K.txt) ──────────────────────────────
    # K.txt: fx=2759.48, fy=2764.16, cx=1520.69, cy=1006.81
    fountain_pts = reconstruct_unknown_cam(
        name="Fountain",
        image_folder=BASE / "Fountain",
        camera_model="PINHOLE",
        camera_params="2759.48,2764.16,1520.69,1006.81",
        match_method='exhaustive',
    )
    save_txt(fountain_pts, OUT / "fountain_colmap_points.txt")
    save_ply(fountain_pts, OUT / "fountain_colmap_points.ply")
    save_viz(fountain_pts, OUT / "fountain_colmap_visualization.png", "Fountain (COLMAP)")
