import xml.etree.ElementTree as ET
import string

def build_sim_xml(src: string, dst: string, start_height: float, kp: float, force: float):
    tree = ET.parse(src)
    root = tree.getroot()

    ET.SubElement(root, "option", timestep="0.002")

    wb = root.find("worldbody")
    base = ET.Element("body", name="base", pos=f"0 0 {start_height}")
    ET.SubElement(base, "freejoint", name="root")
    for child in list(wb):          # move all base geoms and leg bodies under "base"
        wb.remove(child)
        base.append(child)
    wb.append(base)

    # collisions: disable all mesh geoms, then enable only the ankles + floor
    for g in base.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
        if "ankle" in g.get("mesh", ""):
            g.set("contype", "1")
            g.set("conaffinity", "1")
            g.set("friction", "0.7 0.005 0.0001")

    # Add this inside <asset> before the floor geom is created
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                rgb1="1 1 1", rgb2="0.9 0.9 0.9", width="300", height="300")
    ET.SubElement(asset, "material", name="floor_mat", texture="grid",
                texrepeat="5 5", reflectance="0.1")

    ET.SubElement(wb, "geom", name="floor", type="plane", size="2 2 0.1",
                contype="1", conaffinity="1", material="floor_mat")

    # joint damping + position actuators
    act = ET.SubElement(root, "actuator")
    for j in base.iter("joint"):
        if j.get("name") == "root":
            continue
        j.set("damping", "0.5")
        ET.SubElement(act, "position", name=j.get("name") + "_act", joint=j.get("name"), kp=str(kp), forcerange=f"-{force} {force}")

    tree.write(dst)