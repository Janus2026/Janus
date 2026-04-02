import os
import platform
import subprocess
import sys
from typing import Dict

import janus


def launch_janus() -> int:
    system = platform.system()
    binary_name: str = {
        "Linux": "janus",
        # "Windows"
        # "Darwin"
    }.get(system, "janus")

    bin_path: str = os.path.dirname(janus.__file__) + "/janus"

    result = subprocess.run([str(bin_path)] + sys.argv[1:])
    return result.returncode
