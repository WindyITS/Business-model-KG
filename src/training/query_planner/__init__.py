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


def main(argv=None):
    from .dataset import main as _main

    return _main(argv)


__all__ = [
    "SyntheticCompany",
    "SyntheticOffering",
    "SyntheticSegment",
    "build_dataset_manifest",
    "build_dataset_splits",
    "build_synthetic_company_graphs",
    "main",
    "write_dataset_splits",
]
