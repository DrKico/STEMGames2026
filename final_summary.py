"""
Consolidate and summarize final reconstruction results.
Copies the best output for each dataset to clearly-named final files.
"""

import shutil, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

_PROJECT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_PROJECT, "output")
FINAL = os.path.join(_PROJECT, "final_output")
os.makedirs(FINAL, exist_ok=True)

# Best results for each dataset
BEST = {
    "Box":      ("box_points",         "Python ray-triangulation, world coordinates"),
    "Entrance": ("entrance_points",    "Python ray-triangulation, world coordinates"),
    "Statue":   ("statue_colmap_points","COLMAP SfM with bundle adjustment"),
    "Fountain": ("fountain_colmap_points","COLMAP SfM with bundle adjustment"),
}

print("=== Final reconstruction results ===\n")
for name, (stem, method) in BEST.items():
    txt = os.path.join(OUT, f"{stem}.txt")
    ply = os.path.join(OUT, f"{stem}.ply")
    viz = os.path.join(OUT, f"{stem.replace('_colmap','')}_visualization.png")
    if "colmap" in stem:
        viz = os.path.join(OUT, f"{stem.replace('_points','')}_visualization.png")

    pts = np.loadtxt(txt, skiprows=1)
    n = len(pts)

    # Copy to final output
    shutil.copy(txt, os.path.join(FINAL, f"{name.lower()}_final_points.txt"))
    shutil.copy(ply, os.path.join(FINAL, f"{name.lower()}_final_points.ply"))

    print(f"  {name}: {n:,} points  [{method}]")
    if n > 0:
        print(f"    X: [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
        print(f"    Y: [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")
        print(f"    Z: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
    print()

# Multi-panel visualization
fig = plt.figure(figsize=(16, 12))
fig.suptitle("STEM Games 2026 – 3D Point Cloud Reconstruction", fontsize=14, fontweight='bold')

panel = 1
for name, (stem, method) in BEST.items():
    txt = os.path.join(OUT, f"{stem}.txt")
    pts = np.loadtxt(txt, skiprows=1)
    ax = fig.add_subplot(2, 2, panel, projection='3d')
    if len(pts) > 0:
        step = max(1, len(pts) // 15000)
        c = np.clip(pts[::step, 3:6] / 255., 0, 1)
        ax.scatter(pts[::step,0], pts[::step,1], pts[::step,2],
                   c=c, s=0.5, linewidths=0)
    ax.set_title(f"{name}\n{len(pts):,} points")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    panel += 1

plt.tight_layout()
out_viz = os.path.join(FINAL, "all_reconstructions.png")
plt.savefig(out_viz, dpi=150)
plt.close()
print(f"Combined visualization → {out_viz}")
print(f"\nFinal files in: {FINAL}")
