# python python/scripts/export_ref_angles.py assembly/xml/reginleif.xml assembly/json/joint_ref_angles.json

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def export_joint_ref_angles(xml_path: str, output_path: str) -> None:
    path = Path(xml_path)

    if not path.is_file():
        print(f"Error: XML file not found: {path}", file=sys.stderr)
        sys.exit(1)

    root = ET.parse(path).getroot()

    joint_ref_angles = {}

    for joint in root.findall(".//joint"):
        name = joint.get("name", "<unnamed>")
        ref = joint.get("ref")

        joint_ref_angles[name] = float(ref) if ref is not None else None

    # Convert to formatted JSON.
    json_data = json.dumps(joint_ref_angles, indent=2)

    # Print JSON to the terminal.
    print(json_data)

    # Export JSON to a file.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json_data + "\n", encoding="utf-8")

    print(f"\nExported joint reference angles to: {output_file}",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: python {sys.argv[0]} <mujoco_xml_path> "
            "[output_json_path]",
            file=sys.stderr,
        )
        sys.exit(1)

    xml_path = sys.argv[1]

    # Default output filename if one is not provided.
    output_path = (
        sys.argv[2]
        if len(sys.argv) == 3
        else "joint_ref_angles.json"
    )

    export_joint_ref_angles(xml_path, output_path)
