# Environment of the Code Ocean compute capsule for pce-uncertainty-attribution.
#
# The capsule runs the fast path only (unit tests of the PCE core, and every main-text
# figure rebuilt from the committed result files), so it is CPU-only and does not
# install PyTorch. Versions match the paper's reporting summary. If the repository is
# imported into the capsule's /code folder rather than cloned as a capsule, paste this
# file into the capsule's environment editor.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN pip install --upgrade pip && \
    pip install \
        numpy==2.4.3 \
        scipy==1.17.1 \
        matplotlib==3.10.8 \
        pytest==9.0.3
