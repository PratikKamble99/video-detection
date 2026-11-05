"""Shared-memory frame ring with a seqlock.

Why this exists (ARCHITECTURE.md §3): a 1080p BGR frame is 6.2 MB. Pushing it
through ``multiprocessing.Queue`` means pickle → pipe → unpickle, measured at
8-15 ms per frame — up to half the entire 33 ms frame budget, spent moving bytes
you already had. Instead the pixels live in a preallocated shared-memory ring
and only a 40-byte ``FrameRef`` crosses the queue.

There are no locks on the data plane. Correctness comes from a seqlock:

- The producer overwrites the oldest slot and stamps that slot's *current* seq
  into a small shared metadata array.
- A consumer validates the slot's seq immediately before and after copying the
  pixels out. If the producer overwrote the slot mid-read (the consumer fell
  more than ``slots`` frames behind), the two reads disagree and the consumer
  discards the frame rather than returning a torn one.

Never block the producer. If a consumer can't keep up, it loses frames — which
for video is the correct failure (a stale detection is worse than none). Size
the ring at ``slots >= 3 × (max consumer latency / frame period)``.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

from core.events import FrameRef

# Sentinel stored in the meta array for a slot that is empty or mid-write.
# Valid sequence numbers are >= 0, so this never collides with a real frame.
_INVALID = -1


class TornReadError(RuntimeError):
    """Raised by ``read()`` when a slot was overwritten during the copy.

    Not a bug — it is the ring doing its job. The caller fell too far behind and
    should drop this frame and move on to the newest one.
    """


class FrameRing:
    """Preallocated SHM ring. One producer, one-or-more consumers.

    Two shared-memory blocks back it:

    - ``shm``      — the pixel data, shape ``(slots, *frame_shape)``.
    - ``meta_shm`` — one int64 per slot holding the seq currently resident in
                     that slot (or ``_INVALID`` while empty / being written).

    A consumer in another process attaches with ``create=False`` and the two
    names returned by :attr:`names`.
    """

    def __init__(
        self,
        slots: int,
        shape: tuple[int, ...],
        dtype=np.uint8,
        *,
        create: bool = True,
        name: str | None = None,
        meta_name: str | None = None,
    ):
        if slots < 3:
            # < 3 leaves no gap between the slot being written and the slot(s)
            # being read; torn reads become the common case, not the rare one.
            raise ValueError(f"ring needs >= 3 slots, got {slots}")
        self.slots = slots
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

        frame_nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        data_nbytes = slots * frame_nbytes
        meta_nbytes = slots * np.dtype(np.int64).itemsize

        self.shm = shared_memory.SharedMemory(
            create=create, size=data_nbytes, name=name
        )
        self.meta_shm = shared_memory.SharedMemory(
            create=create, size=meta_nbytes, name=meta_name
        )

        self.buf = np.ndarray((slots, *self.shape), dtype=self.dtype, buffer=self.shm.buf)
        self.meta = np.ndarray((slots,), dtype=np.int64, buffer=self.meta_shm.buf)

        if create:
            self.meta[:] = _INVALID

        self._created = create
        self._i = 0  # producer-local write cursor; not shared

    @property
    def names(self) -> tuple[str, str]:
        """(data_name, meta_name) — hand these to a consumer process."""
        return self.shm.name, self.meta_shm.name

    # -- producer side ----------------------------------------------------

    def write(self, frame: np.ndarray, ts_mono_ns: int, seq: int) -> FrameRef:
        """Copy ``frame`` into the next slot and return its reference.

        Ordering matters for the seqlock: we mark the slot invalid, publish the
        pixels, then publish the new seq last. A consumer that reads the seq as
        our ``seq`` is therefore guaranteed to see fully-written pixels.
        """
        if frame.shape != self.shape:
            raise ValueError(f"frame shape {frame.shape} != ring shape {self.shape}")
        slot = self._i % self.slots

        self.meta[slot] = _INVALID          # 1. "this slot is being written"
        self.buf[slot] = frame              # 2. single memcpy, ~1.5 ms @1080p
        self.meta[slot] = seq               # 3. publish: slot now holds `seq`

        self._i += 1
        return FrameRef(
            slot=slot,
            ts_mono_ns=ts_mono_ns,
            seq=seq,
            shape=self.shape,
            dtype=str(self.dtype),
        )

    # -- consumer side ----------------------------------------------------

    def read(self, ref: FrameRef) -> np.ndarray:
        """Return a validated *copy* of the referenced frame.

        Seqlock: check the slot still holds our seq, copy, then re-check. Any
        disagreement means the producer lapped us mid-copy — raise
        ``TornReadError`` and let the caller drop the frame.
        """
        slot = ref.slot
        if self.meta[slot] != ref.seq:
            raise TornReadError(
                f"slot {slot} holds seq {int(self.meta[slot])}, wanted {ref.seq} "
                f"(consumer fell behind before read)"
            )
        out = self.buf[slot].copy()          # snapshot; releases GIL during memcpy
        if self.meta[slot] != ref.seq:
            raise TornReadError(
                f"slot {slot} overwritten to seq {int(self.meta[slot])} during "
                f"read of seq {ref.seq} (torn read)"
            )
        return out

    def read_view(self, ref: FrameRef) -> np.ndarray:
        """Zero-copy view of the slot. Fast path — **unvalidated**.

        Only safe when the caller is provably within ``slots`` frames of the
        producer and finishes using the view before the producer laps it (e.g.
        the reasoning layer's ``zoom_crop`` on a very recent frame). For the
        general consumer, use :meth:`read`.
        """
        return self.buf[ref.slot]

    def is_stale(self, ref: FrameRef) -> bool:
        """True if the slot no longer holds ``ref``'s frame."""
        return bool(self.meta[ref.slot] != ref.seq)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Detach this process from the shared memory (does not free it)."""
        self.shm.close()
        self.meta_shm.close()

    def unlink(self) -> None:
        """Free the shared memory. Call once, from the process that created it."""
        try:
            self.shm.unlink()
            self.meta_shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "FrameRing":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        if self._created:
            self.unlink()
