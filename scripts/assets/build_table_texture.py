"""Build a deterministic mottled green tabletop material; geometry stays referenced."""
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdShade


def main():
    root = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(101)
    noise = np.zeros((512, 512), dtype=np.float32)
    # Periodic Fourier filtering keeps the projected texture seamless when tiled.
    frequency = np.fft.fftfreq(512)
    radius2 = frequency[:, None] ** 2 + frequency[None, :] ** 2
    for sigma, weight in ((20, 0.5), (7, 0.3), (2, 0.2)):
        white = rng.normal(size=(512, 512))
        smooth = np.fft.ifft2(np.fft.fft2(white) * np.exp(-2 * np.pi**2 * sigma**2 * radius2)).real
        noise += weight * smooth / smooth.std() * 0.15
    rgb = np.clip(np.array([49, 85, 59]) + noise[..., None] * 32, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(root / "usd/table_green_texture.png")
    output = root / "usd/table_reference.usda"
    if output.exists():
        stage = Usd.Stage.Open(str(output))
    else:
        stage = Usd.Stage.CreateNew(str(output))
    table = stage.DefinePrim("/green_table", "Xform")
    table.GetReferences().ClearReferences()
    table.GetReferences().AddReference("table.usd", "/green_table")
    stage.SetDefaultPrim(table)
    mat = UsdShade.Material.Define(stage, "/green_table/Materials/reference_green")
    shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    shader.CreateImplementationSourceAttr().Set("sourceAsset")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.09, 0.18, 0.12))
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("table_green_texture.png"))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    top = stage.OverridePrim("/green_table/Geometry/table_base_link/table_top_link/box")
    UsdShade.MaterialBindingAPI.Apply(top).Bind(mat)
    # A kinematic tabletop needs no world-to-static fixed joint in cloned scenes.
    joint = stage.OverridePrim("/green_table/Physics/table_top_joint")
    joint.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    stage.GetRootLayer().Save()
    print(output)


if __name__ == "__main__":
    main()
