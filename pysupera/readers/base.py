"""
Abstract base class for all pysupera foreign-format event readers.

Every reader should subclass :class:`EventReaderBase` and implement
``__len__``, ``__getitem__``, and ``close``.  The base class provides the
context-manager protocol and ``__iter__`` for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..data import Particle


class EventReaderBase(ABC):
    """
    Abstract base class for readers that translate a foreign file format into
    lists of pysupera :class:`~pysupera.data.Particle` objects.

    Concrete subclasses must implement :meth:`__len__`, :meth:`__getitem__`,
    and :meth:`close`.  All other methods are provided by this base.

    The interface is intentionally identical to the native
    :func:`pysupera.io.read_events` store so downstream code (e.g. the
    visualisation app) can consume any source transparently::

        with SomeReader("data.ext") as store:
            particles = store[0]           # one event
            for particles in store:        # iterate all events
                ...
    """

    # ------------------------------------------------------------------ #
    # Context-manager protocol                                            #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "EventReaderBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of events in the file."""

    @abstractmethod
    def __getitem__(self, index: int) -> list[Particle]:
        """
        Return the list of :class:`~pysupera.data.Particle` objects for
        event *index* (0-based).
        """

    @abstractmethod
    def close(self) -> None:
        """Release any open file handles or other resources."""

    # ------------------------------------------------------------------ #
    # Provided                                                             #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[list[Particle]]:
        """Iterate over all events in order."""
        for i in range(len(self)):
            yield self[i]

    def iter_events(self) -> Iterator[list[Particle]]:
        """Alias for :meth:`__iter__`; mirrors the :class:`~pysupera.io.EventStore` API."""
        return iter(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_events={len(self)})"
