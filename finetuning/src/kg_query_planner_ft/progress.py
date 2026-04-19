from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent
    tqdm = None


T = TypeVar("T")


def progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
        return
    print(message)


def track(
    items: Iterable[T],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
) -> Iterable[T]:
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc, unit=unit, dynamic_ncols=True)


@dataclass
class StepProgress(AbstractContextManager["StepProgress"]):
    total: int
    desc: str
    unit: str = "step"

    def __post_init__(self) -> None:
        self._current = 0
        self._bar = None
        if tqdm is not None:
            self._bar = tqdm(total=self.total, desc=self.desc, unit=self.unit, dynamic_ncols=True)

    def __enter__(self) -> "StepProgress":
        return self

    def advance(self, message: str | None = None) -> None:
        self._current += 1
        if self._bar is not None:
            if message:
                self._bar.set_postfix_str(message)
            self._bar.update(1)
            return
        if message:
            progress_write(f"[{self._current}/{self.total}] {self.desc}: {message}")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        self.close()
