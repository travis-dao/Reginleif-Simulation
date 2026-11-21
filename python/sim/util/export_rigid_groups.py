# python python/scripts/export_rigid_groups.py assembly/urdf/assembly.urdf assembly/json/rigid_groups.json

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def export_rigid_groups(urdf_path: str, output_path: str) -> None:
    path = Path(urdf_path)

    if not path.is_file():
        sys.exit(f"Error: URDF file not found: {path}")

    root = ET.parse(path).getroot()

    total_mass = 0.0
    links = root.findall("link")
    rigid_groups = []

    print(f"{'Rigid Group (Link)':<30} {'Mass (kg)':>12}")
    print("-" * 44)

    for link in links:
        name = link.get("name", "<unnamed>")
        inertial = link.find("inertial")
        mass_element = (
            inertial.find("mass") if inertial is not None else None
        )

        if mass_element is not None:
            mass = float(mass_element.get("value", "0"))
            total_mass += mass
            mass_value = mass
            mass_str = f"{mass:.6f}"
        else:
            mass_value = None
            mass_str = "N/A"

        rigid_groups.append({
            "name": name,
            "mass_kg": mass_value,
        })

        print(f"{name:<30} {mass_str:>12}")

    print("-" * 44)
    print(f"{'Total known mass':<30} {total_mass:>12.6f} kg")
    print(f"Number of links: {len(links)}")

    # Assemble JSON output.
    data = {
        "rigid_groups": rigid_groups,
        "total_known_mass_kg": total_mass,
        "number_of_links": len(links),
    }

    json_data = json.dumps(data, indent=2)

    # Export JSON.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json_data + "\n", encoding="utf-8")

    print(f"\nExported rigid group data to: {output_file}",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (1, 2, 3):
        sys.exit(
            f"Usage: python {sys.argv[0]} "
            "[urdf_path] [output_json_path]"
        )

    urdf_path = (
        sys.argv[1] if len(sys.argv) >= 2 else "assembly.urdf"
    )

    output_path = (
        sys.argv[2]
        if len(sys.argv) == 3
        else "rigid_groups.json"
    )

    export_rigid_groups(urdf_path, output_path)