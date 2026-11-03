"""Low-poly copies of the robot's visual meshes, for fast GPU ray tracing.

The Warp ray tracer slows down with triangle count: the 322k-triangle visual
meshes render at ~550 frames/s (424x240, 64 worlds), the low-poly ones at
~950. At 424x240 the two images look the same.

Only VISUAL meshes are swapped (contype = conaffinity = 0). Collision geoms
and every body's mass/inertia (explicit <inertial> in robot.xml) are
untouched, so the physics is identical.

Vertex clustering is used rather than quadric decimation: these STL parts
have many open borders and quadric decimation stalls at ~30%.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from vision_rl.envs import ASSETS_DIR, SCENE_XML

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vision_rl", "lowpoly")


def _cluster(vertices: np.ndarray, faces: np.ndarray, cell: float):
    """Snap vertices to a grid of `cell` size, merge each cell, drop flat faces."""
    key = np.floor(vertices / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    count = np.bincount(inv)
    v = np.zeros((count.size, 3))
    np.add.at(v, inv, vertices)
    v /= count[:, None]
    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return v, f[keep]


def visual_mesh_names(model: mujoco.MjModel) -> set[str]:
    names = set()
    for i in range(model.ngeom):
        if (model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                and model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0):
            names.add(model.mesh(model.geom_dataid[i]).name)
    return names


def load_lowpoly_model(xml: str = SCENE_XML, cell_mm: float = 1.5) -> mujoco.MjModel:
    """The scene with its visual meshes replaced by cached low-poly copies."""
    import trimesh  # only needed the first time a cache is written

    full = mujoco.MjModel.from_xml_path(xml)
    visual = visual_mesh_names(full)
    spec = mujoco.MjSpec.from_file(xml)
    mesh_dir = os.path.join(ASSETS_DIR, "meshes")
    out_dir = os.path.join(CACHE_DIR, f"cell{cell_mm:g}mm")
    os.makedirs(out_dir, exist_ok=True)
    for mesh in spec.meshes:
        if mesh.name not in visual:
            continue
        src = os.path.join(mesh_dir, mesh.file)
        dst = os.path.join(out_dir, os.path.basename(mesh.file))
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            t = trimesh.load(src, force="mesh")
            scale = 1000.0 if t.extents.max() > 1.0 else 1.0     # STL in mm or m
            v, f = _cluster(np.asarray(t.vertices), np.asarray(t.faces), cell_mm / 1000.0 * scale)
            trimesh.Trimesh(v, f, process=False).export(dst + ".tmp.stl")
            os.replace(dst + ".tmp.stl", dst)
        mesh.file = dst
    spec.meshdir = mesh_dir
    return spec.compile()
