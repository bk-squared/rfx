"""Isolated two-host-device matrix worker; environment is set by the parent."""
import json
import sys
import traceback

from .builders import BUILDERS
from .execution import execute
from .generation import generate


def main():
    cells = {cell.id: cell for cell in generate(BUILDERS)}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            report = execute(cells[request['cell']])
            print(json.dumps(report), flush=True)
        except Exception:
            print(json.dumps({'worker_error': traceback.format_exc()}), flush=True)


if __name__ == '__main__':
    main()
