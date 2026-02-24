"""Bulk-create Daytona snapshots for all tasks in an ARES preset.

Snapshots eliminate the declarative build step when creating sandboxes, replacing
``CreateSandboxFromImageParams`` with the much faster ``CreateSandboxFromSnapshotParams``.

Usage:
    uv run python -m ares.tinker_integration.create_snapshots \
        --preset sbv-terminus2 \
        --template "ares__{name}" \
        --num-tasks 20 \
        --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import daytona
import daytona.common.errors

from ares.tinker_integration import dataset

_LOGGER = logging.getLogger(__name__)


async def _create_single_snapshot(
    client: daytona.AsyncDaytona,
    *,
    snapshot_name: str,
    task: Any,
    force_recreate: bool,
    semaphore: asyncio.Semaphore,
) -> str:
    """Create a single snapshot, returning a status string."""
    async with semaphore:
        # Check if snapshot already exists.
        if not force_recreate:
            try:
                existing = await client.snapshot.get(snapshot_name)
                if existing.state == "ACTIVE":
                    _LOGGER.info("SKIP (active): %s", snapshot_name)
                    return "skipped"
            except daytona.common.errors.DaytonaNotFoundError:
                pass

        # Determine image.
        docker_image = task.config.environment.docker_image
        if docker_image is not None:
            image = daytona.Image.base(docker_image)
        else:
            dockerfile_path = task.paths.environment_dir / "Dockerfile"
            image = daytona.Image.from_dockerfile(str(dockerfile_path))

        # Build resources from task config.
        resources = daytona.Resources(
            cpu=task.config.environment.cpus,
            memory=task.config.environment.memory_mb // 1024,
            disk=task.config.environment.storage_mb // 1024,
        )

        _LOGGER.info("CREATE: %s (image=%s)", snapshot_name, docker_image or "Dockerfile")
        try:
            params = daytona.CreateSnapshotParams(
                name=snapshot_name,
                image=image,
                resources=resources,
            )
            await client.snapshot.create(params)
            _LOGGER.info("OK: %s", snapshot_name)
            return "created"
        except Exception as e:
            _LOGGER.error("FAIL: %s — %s: %s", snapshot_name, type(e).__name__, e)
            return "failed"


async def create_snapshots(
    *,
    preset_name: str,
    template: str,
    num_tasks: int | None = None,
    concurrency: int = 5,
    force_recreate: bool = False,
) -> dict[str, int]:
    """Create Daytona snapshots for tasks in a preset.

    Args:
        preset_name: ARES preset name (e.g., "sbv-terminus2").
        template: Snapshot name template with ``{name}`` placeholder.
        num_tasks: Limit tasks (None = all).
        concurrency: Max concurrent snapshot creations.
        force_recreate: Re-create even if an active snapshot exists.

    Returns:
        Summary dict with counts: created, skipped, failed.
    """
    if "{name}" not in template:
        raise ValueError("template must contain '{name}' placeholder")

    tasks = dataset.load_tasks_from_preset(preset_name, num_tasks=num_tasks)
    _LOGGER.info("Creating snapshots for %d tasks from preset '%s'", len(tasks), preset_name)

    client = daytona.AsyncDaytona()
    semaphore = asyncio.Semaphore(concurrency)

    coros = [
        _create_single_snapshot(
            client,
            snapshot_name=template.format(name=task.name),
            task=task,
            force_recreate=force_recreate,
            semaphore=semaphore,
        )
        for task in tasks
    ]

    results = await asyncio.gather(*coros, return_exceptions=True)

    summary: dict[str, int] = {"created": 0, "skipped": 0, "failed": 0}
    for r in results:
        if isinstance(r, str):
            summary[r] = summary.get(r, 0) + 1
        else:
            _LOGGER.error("Unexpected error: %s", r)
            summary["failed"] += 1

    _LOGGER.info("Snapshot creation summary: %s", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Bulk-create Daytona snapshots for ARES tasks")
    p.add_argument("--preset", type=str, required=True, help="ARES preset name (e.g., sbv-terminus2)")
    p.add_argument("--template", type=str, required=True, help="Snapshot name template with {name} placeholder")
    p.add_argument("--num-tasks", type=int, default=None, help="Limit number of tasks")
    p.add_argument("--concurrency", type=int, default=5, help="Max concurrent snapshot creations")
    p.add_argument("--force-recreate", action="store_true", help="Re-create even if active snapshot exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    asyncio.run(
        create_snapshots(
            preset_name=args.preset,
            template=args.template,
            num_tasks=args.num_tasks,
            concurrency=args.concurrency,
            force_recreate=args.force_recreate,
        )
    )


if __name__ == "__main__":
    main()
