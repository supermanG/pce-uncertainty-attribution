# Code Ocean capsule

The compute capsule is this repository imported into `/code` (Code Ocean, Create New >
Capsule > Copy from Public Git, URL `https://github.com/supermanG/pce-uncertainty-attribution`).
The entry point is `/code/run`, the `run` script at the repository root: it runs the unit tests
and rebuilds the six main-text figures from the committed result files, writing them and a
SHA-256 list to `/results`.

- `environment/Dockerfile`: the capsule environment (python:3.14-slim, numpy 2.4.3,
  scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, pytest 9.0.3; CPU only). Set it in
  the capsule's environment editor, or pick a Python starter environment and add the five pip
  packages (the sparse-PCE unit test imports scikit-learn).
- `metadata/metadata.yml`: the capsule record (title, description, tags, authors and
  affiliations), to be entered in the capsule's metadata panel.

These two folders sit under `codeocean/` on purpose: at the repository root they make the
importer treat the repository as a capsule layout, leaving `/code` empty.
