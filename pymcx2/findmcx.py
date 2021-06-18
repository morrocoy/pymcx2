import sys
import pathlib


def find_mcx():
    for path in sys.path:
        glob_path = pathlib.Path(path)
        patterns = ["bin/mcx.exe", "mcx.exe", "bin/mcx", "mcx"]
        for pattern in patterns:
            for pp in glob_path.glob(pattern):
                return str(pp)

    return None
