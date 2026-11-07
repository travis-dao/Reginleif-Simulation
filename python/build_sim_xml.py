# python python/scripts/build_sim_xml.py

import xml.etree.ElementTree as ET
import math
import json

with open('assembly/json/joint_ref_angles.json', 'r') as file:
    REF_ANGLES_DEG = json.load(file)

def build_sim_xml(
    src: str,
    dst: str,
    start_height: float,
    kp: float,
    force: float,
):
    tree = ET.parse(src)
    root = tree.getroot()

    # Ensure joint reference angles use radians in the generated MJCF.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")

    ET.SubElement(root, "option", timestep="0.002")

    wb = root.find("worldbody")

    # Free body: identity orientation, so yaw == 0 means +X.
    base = ET.Element("body", name="base", pos=f"0 0 {start_height}")
    ET.SubElement(base, "freejoint", name="root")

    # Rotate the robot +90 degrees about Z.
    rot = ET.SubElement(
        base,
        "body",
        name="base_rot",
        pos=f"0 0 0",
        quat="0.707107 0 0 0.707107",
    )

    for child in list(wb):
        wb.remove(child)
        rot.append(child)
    wb.append(base)

    # Set reference angles and joint damping.
    found_joints = set()

    for j in base.iter("joint"):
        name = j.get("name")

        if name == "root":
            continue

        if name in REF_ANGLES_DEG:
            angle_deg = REF_ANGLES_DEG[name]
            angle_rad = math.radians(angle_deg)
            j.set("ref", str(angle_rad))
            found_joints.add(name)

            # print(f"{name}: ref = {angle_deg} deg ({angle_rad:.6f} rad)")

        j.set("damping", "0.5")

    missing_joints = set(REF_ANGLES_DEG) - found_joints
    if missing_joints:
        raise ValueError(
            f"Could not find configured joints in {src}: "
            f"{sorted(missing_joints)}"
        )

    # Collisions: disable all mesh geoms except ankles.
    for g in base.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")

        if "ankle" in g.get("mesh", ""):
            g.set("contype", "1")
            g.set("conaffinity", "1")
            g.set("friction", "0.7 0.005 0.0001")

    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="1 1 1",
        rgb2="0.9 0.9 0.9",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_mat",
        texture="grid",
        texrepeat="5 5",
        reflectance="0.1",
    )

    ET.SubElement(
        wb,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        contype="1",
        conaffinity="1",
        material="floor_mat",
    )

    # Position actuators.
    act = ET.SubElement(root, "actuator")
    for j in base.iter("joint"):
        name = j.get("name")
        if name == "root":
            continue

        ET.SubElement(
            act,
            "position",
            name=name + "_act",
            joint=name,
            kp=str(kp),
            forcerange=f"-{force} {force}",
        )

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
