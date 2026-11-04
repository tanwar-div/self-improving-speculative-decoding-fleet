"""Single entrypoint for every process in the fleet."""

from __future__ import annotations

import logging
import os
import sys

import typer
import uvicorn

from fleet.config import settings

cli = typer.Typer(add_completion=False, help="Self-improving speculative decoding fleet")


def _logging() -> None:
    logging.basicConfig(
        level=os.environ.get("FLEET_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@cli.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run a serving replica."""
    _logging()
    uvicorn.run("fleet.serving.app:app", host=host, port=port, workers=1, log_level="info")


@cli.command()
def router(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the fleet router."""
    _logging()
    uvicorn.run("fleet.router.app:app", host=host, port=port, workers=1, log_level="info")


@cli.command()
def trainer() -> None:
    """Run the asynchronous draft-model trainer."""
    _logging()
    from fleet.trainer.worker import run_forever

    run_forever()


@cli.command()
def controller(interval: float = 5.0) -> None:
    """Run the canary rollout controller."""
    _logging()
    from fleet.canary.controller import run_forever

    run_forever(interval=interval)


@cli.command()
def bootstrap() -> None:
    """Apply the schema and register a cold-start version."""
    _logging()
    from fleet import db
    from fleet.trainer.worker import bootstrap as _bootstrap

    db.wait_for_db()
    db.apply_schema()
    version = _bootstrap()
    typer.echo(f"active version: v{version}")


@cli.command()
def status() -> None:
    """Print registry state and recent deployment history."""
    _logging()
    import json

    from fleet.registry import store

    dep = store.get_deployment()
    typer.echo(
        json.dumps(
            {
                "deployment": dep.__dict__,
                "versions": [v.__dict__ for v in store.list_versions(10)],
                "history": store.deployment_history(15),
            },
            indent=2,
            default=str,
        )
    )


@cli.command("build-prompts")
def build_prompts(total: int = 40_000, force: bool = False) -> None:
    """Build the multi-domain prompt pool."""
    _logging()
    import json

    from fleet.bench.prompts import build_pool

    _, manifest = build_pool(total=total, force=force)
    typer.echo(json.dumps(manifest, indent=2))

    # The `datasets`/pyarrow stack can abort during interpreter teardown
    # ("PyGILState_Release: thread state must be current") from its background
    # reader threads -- long after the pool has been written and the manifest
    # printed. Exiting here keeps a successful build from reporting failure,
    # which would otherwise break `make bench`, since that target depends on
    # this one. Everything this command writes is already flushed to disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@cli.command()
def info() -> None:
    """Print the effective configuration."""
    typer.echo(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    cli()
