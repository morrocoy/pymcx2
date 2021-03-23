import sys
import pathlib

def findMCX():
    for path in sys.path:
        glob_path = pathlib.Path(path)
        for pp in glob_path.glob("bin/mcx.exe"):
            return pp

    return None
