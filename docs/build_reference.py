"""Build the jury's downloadable Sphinx HTML and OpenAPI snapshots from repository code.

Run from the repository root with the project virtual environment. No services, models,
training, or private environment files are required to generate the API specifications.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "docs/sphinx/_build/html"
    if output.exists():
        shutil.rmtree(output)  # Only our generated HTML; avoid stale pages in the release ZIP.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-E",
            "-b",
            "html",
            str(ROOT / "docs/sphinx"),
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    from transport_backend.api import app as backend_app
    from transport_ml.service import app as ml_app

    api_dir = ROOT / "docs/openapi"
    api_dir.mkdir(exist_ok=True)
    for name, app in [("backend", backend_app), ("ml", ml_app)]:
        (api_dir / f"{name}.json").write_text(
            json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance = {
        "source_commit": commit,
        "source_tree_note": "Built from the worktree based on source_commit.",
        "command": ".venv/bin/python docs/build_reference.py",
        "entrypoint": "index.html",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    archive = ROOT / "docs/code-reference.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and ".doctrees" not in path.parts and path.name != ".buildinfo":
                bundle.write(path, path.relative_to(output))
        for path in sorted(api_dir.glob("*.json")):
            bundle.write(path, f"openapi/{path.name}")
    print(f"Generated {archive.relative_to(ROOT)} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
