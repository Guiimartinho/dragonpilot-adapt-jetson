import os
from typing import cast

from openpilot.system.hardware.base import HardwareBase
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.pc.hardware import Pc

TICI = os.path.isfile('/TICI')
AGNOS = os.path.isfile('/AGNOS')
JETSON = os.path.isfile('/JETSON')
PC = not TICI and not JETSON


if TICI:
  HARDWARE = cast(HardwareBase, Tici())
elif JETSON:
  from openpilot.system.hardware.jetson.hardware import Jetson
  HARDWARE = cast(HardwareBase, Jetson())
else:
  HARDWARE = cast(HardwareBase, Pc())
