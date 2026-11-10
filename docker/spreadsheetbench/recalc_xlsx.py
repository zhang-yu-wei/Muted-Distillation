"""Refresh cached formula values of .xlsx files in place via headless LibreOffice.

Agent-written workbooks (openpyxl saves formulas without cached results) must be
recalculated before the official SpreadsheetBench v1/v2 comparison, which reads
computed values with ``openpyxl.load_workbook(data_only=True)``. This is the
same load -> calculateAll -> store cycle as the benchmark's
``evaluation/open_spreadsheet.py``, reduced to a one-shot CLI.

Run inside the sandbox image with the interpreter that owns the UNO bindings:

    /usr/bin/python3 /opt/recalc_xlsx.py file1.xlsx [file2.xlsx ...]

Exits non-zero if any file fails; per-file status lines go to stdout.
"""

import os
import subprocess
import sys
import time

os.environ.setdefault(
    "URE_BOOTSTRAP",
    "vnd.sun.star.pathname:/usr/lib/libreoffice/program/fundamentalrc",
)

import uno  # noqa: E402  (needs URE_BOOTSTRAP set first)
from com.sun.star.beans import PropertyValue  # noqa: E402

UNO_PORT = 2002
SERVICE_READY_TIMEOUT_SEC = 60


def connect_desktop():
    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    deadline = time.time() + SERVICE_READY_TIMEOUT_SEC
    while True:
        try:
            context = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={UNO_PORT};urp;"
                "StarOffice.ComponentContext"
            )
            break
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(1)
    return context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )


def recalc_file(desktop, path):
    path = os.path.abspath(path)
    temp_output = path + ".recalc.xlsx"
    doc = desktop.loadComponentFromURL(f"file://{path}", "_blank", 0, ())
    if doc is None:
        raise RuntimeError("LibreOffice failed to open the document")
    try:
        # Match the official evaluator's settings: circular references in
        # financial models need iterative calculation to converge.
        doc.IsIterationEnabled = True
        doc.IterationCount = 100
        doc.IterationEpsilon = 0.0001
        doc.calculateAll()
        save_props = (
            PropertyValue(Name="FilterName", Value="Calc MS Excel 2007 XML"),
        )
        doc.storeToURL(f"file://{temp_output}", save_props)
    finally:
        doc.close(True)
    os.replace(temp_output, path)


def main(paths):
    soffice = subprocess.Popen(
        [
            "soffice",
            "--headless",
            # Explicit profile dir: without it, profile setup fails with
            # "User installation could not be completed" when the container
            # runs as a non-root --user (as the eval harness does).
            "-env:UserInstallation=file:///tmp/lo_profile_recalc",
            f"--accept=socket,host=127.0.0.1,port={UNO_PORT};urp;",
            "--norestore",
            "--nofirststartwizard",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    failures = 0
    try:
        desktop = connect_desktop()
        for path in paths:
            try:
                recalc_file(desktop, path)
                print(f"OK {path}")
            except Exception as error:
                failures += 1
                print(f"FAIL {path}: {error}")
            sys.stdout.flush()
    finally:
        soffice.terminate()
        try:
            soffice.wait(timeout=10)
        except subprocess.TimeoutExpired:
            soffice.kill()
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
