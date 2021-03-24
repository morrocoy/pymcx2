import sys
import pathlib

def findMCX():
    for path in sys.path:
        glob_path = pathlib.Path(path)
        patterns = ["bin/mcx.exe", "mcx.exe"]
        for pattern in patterns:
            for pp in glob_path.glob(pattern):
                return pp

    return None
