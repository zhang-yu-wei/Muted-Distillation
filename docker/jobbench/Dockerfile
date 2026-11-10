# Sandbox image for JobBench episodes.
#
# JobBench tasks read CSV/XLSX/DOCX/PDF/SQLite reference files and write
# finished deliverables (usually .xlsx, .docx, .md, .csv, sometimes .pdf or a
# chart image), so the image carries a reader and a writer for each of those
# formats. The container runs `sleep` and takes commands over `docker exec`,
# like the SpreadsheetBench sandbox.
#
# Build: docker build -t jobbench-sandbox docker/jobbench
#
# LibreOffice is deliberately absent: it adds ~500 MB and ~2 s to every boot,
# and nothing here needs format conversion or formula recalculation — the judge
# reads deliverables with the same pure-python libraries listed below. Add
# `libreoffice-calc libreoffice-writer` here if a task set starts needing
# conversion fidelity.
FROM python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# fonts-dejavu-core: matplotlib figures render without font warnings.
# poppler-utils: pdftotext, a fast fallback when pdfplumber is slow on a file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        poppler-utils \
        sqlite3 \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        matplotlib \
        numpy \
        odfpy \
        openpyxl \
        pandas \
        pdfplumber \
        pypdf \
        python-docx \
        python-pptx \
        reportlab \
        tabulate \
        xlsxwriter

WORKDIR /workspace
