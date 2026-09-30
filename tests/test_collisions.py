"""
Audits all geoms in the MuJoCo model to ensure collision properties are properly enabled.
"""
import mujoco


def audit_collisions():
    model = mujoco.MjModel.from_xml_path("models/mujoco/scene.xml")
    print(f"Total Geoms in Model: {model.ngeom}\n")

    collision_geoms = []
    visual_only_geoms = []

    for gid in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

        contype = model.geom_contype[gid]
        conaffinity = model.geom_conaffinity[gid]
        geom_type = model.geom_type[gid]

        is_collidable = (contype != 0) or (conaffinity != 0)

        entry = (gid, geom_name, body_name, contype, conaffinity, is_collidable)
        if is_collidable:
            collision_geoms.append(entry)
        else:
            visual_only_geoms.append(entry)

    print(f"✅ Active Collision Geoms: {len(collision_geoms)}")
    for gid, name, body, ct, ca, _ in collision_geoms:
        print(f"  [ID {gid:2d}] {name:<25} | Body: {body:<22} | contype={ct} conaffinity={ca}")

    print(f"\n👁️  Visual-Only Geoms (contype=0, conaffinity=0): {len(visual_only_geoms)}")
    for gid, name, body, ct, ca, _ in visual_only_geoms[:10]:
        print(f"  [ID {gid:2d}] {name:<25} | Body: {body:<22}")
    if len(visual_only_geoms) > 10:
        print(f"  ... and {len(visual_only_geoms) - 10} more.")

    # Key objects check
    critical_bodies = ["table", "cube", "target_box", "moving_jaw_so101_v1", "gripper"]
    print("\nCritical Object Collision Status:")
    for bname in critical_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            print(f"  ❌ Missing body: {bname}")
            continue
        has_col = any(model.geom_bodyid[gid] == bid and ((model.geom_contype[gid] != 0) or (model.geom_conaffinity[gid] != 0)) for gid in range(model.ngeom))
        print(f"  {'✅' if has_col else '❌'} {bname:<25}: {'Collidable' if has_col else 'NO COLLISION'}")


if __name__ == "__main__":
    audit_collisions()
