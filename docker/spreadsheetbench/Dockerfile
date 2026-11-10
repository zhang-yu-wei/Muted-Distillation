# Sandbox image for SpreadsheetBench v1 and v2 evaluation.
#
# Mirrors the official environment
# (third_party/SpreadsheetBench-2/SWE-agent/spreadsheet.Dockerfile): python 3.11
# with openpyxl/pandas/numpy/matplotlib/xlsxwriter plus LibreOffice. The
# official image's swe-rex runtime is omitted because our harness drives the
# container over `docker exec`.
#
# Build: docker build -t spreadsheetbench-v1-sandbox docker/spreadsheetbench
# (The v2 launcher builds the same context under spreadsheetbench-v2-sandbox.)
FROM python:3.11-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN pip install --no-cache-dir openpyxl numpy pandas matplotlib xlsxwriter

RUN apt-get update \
    && apt-get install -y libreoffice libreoffice-script-provider-python \
    && rm -rf /var/lib/apt/lists/*

# The UNO bindings from libreoffice-script-provider-python belong to the
# system interpreter (/usr/bin/python3), not the image's python 3.11. Keeping
# both locations on PYTHONPATH matches the official image; recalc_xlsx.py must
# be run with /usr/bin/python3.
ENV PYTHONPATH="/usr/lib/libreoffice/program:/usr/lib/python3/dist-packages"

# Same read-only inspection tool the official SWE-agent harness exposes.
COPY view_xlsx /usr/local/bin/view_xlsx
RUN chmod +x /usr/local/bin/view_xlsx

COPY recalc_xlsx.py /opt/recalc_xlsx.py
