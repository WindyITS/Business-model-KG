"""Training-side dataset tooling for the query planner."""

from .graphs import SyntheticCompany, SyntheticOffering, SyntheticSegment, build_synthetic_company_graphs


def build_dataset_manifest(*args, **kwargs):
    from .dataset import build_dataset_manifest as _build_dataset_manifest

    return _build_dataset_manifest(*args, **kwargs)


def build_dataset_splits(*args, **kwargs):
    from .dataset import build_dataset_splits as _build_dataset_splits

    return _build_dataset_splits(*args, **kwargs)


def write_dataset_splits(*args, **kwargs):
    from .dataset import write_dataset_splits as _write_dataset_splits

    return _write_dataset_splits(*args, **kwargs)


def build_curated_artifact(*args, **kwargs):
    from .curated_artifact import build_curated_artifact as _build_curated_artifact

    return _build_curated_artifact(*args, **kwargs)


def freeze_curated_baseline(*args, **kwargs):
    from .curated_artifact import freeze_curated_baseline as _freeze_curated_baseline

    return _freeze_curated_baseline(*args, **kwargs)


def verify_curated_artifact(*args, **kwargs):
    from .curated_artifact import verify_curated_artifact as _verify_curated_artifact

    return _verify_curated_artifact(*args, **kwargs)


def main(argv=None):
    from .dataset import main as _main

    return _main(argv)


__all__ = [
    "SyntheticCompany",
    "SyntheticOffering",
    "SyntheticSegment",
    "build_curated_artifact",
    "build_dataset_manifest",
    "build_dataset_splits",
    "build_synthetic_company_graphs",
    "freeze_curated_baseline",
    "main",
    "verify_curated_artifact",
    "write_dataset_splits",
]
