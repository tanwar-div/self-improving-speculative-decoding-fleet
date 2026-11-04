"""Guard against a failure class the host test suite cannot see.

`scipy` was missing from `requirements-container.txt`. Every host test passed,
the image built, the containers started healthy -- and the canary controller
then crashed on every single tick inside its per-tick exception handler. The
fleet served 241 requests with a candidate stuck in canary and not one SLO
evaluation recorded, which reads exactly like "the canary is still undecided"
rather than like a missing dependency.

So: every third-party package imported anywhere under `src/fleet` must be
declared for the container image, because one image runs every role.
"""

from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "fleet"
REQS = REPO / "requirements-container.txt"
DOCKERFILE = REPO / "Dockerfile"

# Import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "psycopg": "psycopg",
    "psycopg_pool": "psycopg",
    "pydantic_settings": "pydantic-settings",
    "scipy": "scipy",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}

# Imported only by code that never runs inside the serving image.
CONTAINER_EXEMPT = {
    "datasets",  # the prompt pool is built on the host, not in a replica
}


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _strip_spec(token: str) -> str:
    for sep in (">=", "==", "~=", "!=", ">", "<", "["):
        if sep in token:
            token = token.split(sep, 1)[0]
    return token.strip().lower()


def _declared_distributions() -> set[str]:
    """Everything the image installs: the requirements file plus any package
    named directly on a `pip install` line in the Dockerfile.

    torch, for instance, is installed on its own line from the CPU wheel index
    so that the large layer caches independently of application dependency
    churn -- it is in the image without being in the requirements file.
    """
    out: set[str] = set()
    for raw in REQS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.add(_strip_spec(line))

    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "pip install" not in line:
            continue
        tokens = line.split()
        skip_next = False
        for tok in tokens[tokens.index("install") + 1 :]:
            if skip_next:
                skip_next = False
                continue
            if tok in {"--index-url", "--extra-index-url", "-r", "--find-links"}:
                skip_next = True
                continue
            if tok.startswith("-") or tok == "\\":
                continue
            out.add(_strip_spec(tok))
    return out


def test_every_third_party_import_is_declared_for_the_container():
    stdlib = set(sys.stdlib_module_names)
    declared = _declared_distributions()

    missing: dict[str, set[str]] = {}
    for py in sorted(SRC.rglob("*.py")):
        for name in _top_level_imports(py):
            if name in stdlib or name == "fleet" or name in CONTAINER_EXEMPT:
                continue
            dist = IMPORT_TO_DIST.get(name, name).lower()
            if dist not in declared:
                missing.setdefault(dist, set()).add(str(py.relative_to(REPO)))

    assert not missing, (
        "third-party imports under src/fleet that requirements-container.txt "
        f"does not declare: { {k: sorted(v) for k, v in missing.items()} }"
    )


def test_exempt_modules_really_are_absent_from_the_serving_path():
    """An exemption must stay true. If a serving module starts importing
    `datasets`, the exemption silently becomes a lie."""
    serving_roots = ["serving", "router", "canary", "registry", "specdec", "drift", "trainer"]
    offenders: dict[str, set[str]] = {}
    for sub in serving_roots:
        for py in sorted((SRC / sub).rglob("*.py")):
            for name in _top_level_imports(py):
                if name in CONTAINER_EXEMPT:
                    offenders.setdefault(name, set()).add(str(py.relative_to(REPO)))
    assert not offenders, (
        f"modules exempted from the container image are imported on the serving path: "
        f"{ {k: sorted(v) for k, v in offenders.items()} }"
    )
