# mjpython python/scripts/urdf_to_xml.py

import mujoco

# Load URDF directly into MuJoCo model
model = mujoco.MjModel.from_xml_path("assembly/urdf/assembly.urdf")

# Save the loaded model as an MJCF XML file
mujoco.mj_saveLastXML("assembly/xml/reginleif.xml", model)