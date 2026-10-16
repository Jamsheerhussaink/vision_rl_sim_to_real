"""Visual-only cube grain; collision geometry remains the original cuboid."""
from pathlib import Path


def spawn_grain_material(prim_path, cfg, **kwargs):
    return _spawn_textured(prim_path, cfg, "cube_grain.png", **kwargs)


def spawn_cardboard_material(prim_path, cfg, **kwargs):
    return _spawn_textured(prim_path, cfg, "box_grain.png", **kwargs)


def _spawn_textured(prim_path, cfg, filename, **kwargs):
    from isaaclab.sim.spawners.materials.visual_materials import spawn_from_mdl_file
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Sdf, UsdShade

    prim = spawn_from_mdl_file(prim_path, cfg, **kwargs)
    shader = UsdShade.Shader(get_current_stage().GetPrimAtPath(prim_path + "/Shader"))
    texture = Path(__file__).resolve().parents[3] / "usd" / filename
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    scale = (20.0, 20.0) if "cube" in filename else (2.0, 2.0)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(scale)
    return prim
