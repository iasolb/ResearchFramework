"""Make the test suite test THIS checkout, not whatever is pip-installed.

This is a src-layout package, so `import otter` normally resolves
to the installed distribution. If an editable install points somewhere else (a
second clone, a scratch directory), the suite silently tests that other tree
and a green run says nothing about the code you are looking at. That was the
live situation in the sibling Census_Loader repo on 2026-08-28.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
