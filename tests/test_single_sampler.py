"""Tests for terratorch.samplers.single to maximize coverage.

Covers:
- MultiSampler length aggregation
- RandomMultiSampler iteration logic, removal on StopIteration
- SequentialMultiSampler ordered yielding
- Edge cases: empty sampler list, single-item samplers, mixed lengths
- Deterministic path via patching random.choice
- Works with stubbed torchgeo if dependency absent
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from random import choice as real_choice
import random
import pytest

# ---- Provide torchgeo stubs if library not available ----
try:
    import torchgeo  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only if torchgeo missing
    # datasets submodule with BoundingBox
    datasets_mod = types.ModuleType('torchgeo.datasets')
    @dataclass(frozen=True)
    class BoundingBox:
        minx: float = 0.0
        maxx: float = 1.0
        miny: float = 0.0
        maxy: float = 1.0
        mint: float = 0.0
        maxt: float = 1.0
    datasets_mod.BoundingBox = BoundingBox
    # samplers submodule with GeoSampler base
    samplers_mod = types.ModuleType('torchgeo.samplers')
    class GeoSampler:  # minimal base
        pass
    samplers_mod.GeoSampler = GeoSampler
    # register modules
    sys.modules['torchgeo.datasets'] = datasets_mod
    sys.modules['torchgeo.samplers'] = samplers_mod
    # create parent torchgeo package if missing
    if 'torchgeo' not in sys.modules:
        torchgeo_pkg = types.ModuleType('torchgeo')
        sys.modules['torchgeo'] = torchgeo_pkg

from torchgeo.datasets import BoundingBox
from torchgeo.samplers import GeoSampler

# Import module under test
from terratorch.samplers.single import MultiSampler, RandomMultiSampler, SequentialMultiSampler

# ---- Helper dummy sampler implementations ----
class DummySampler(GeoSampler):
    def __init__(self, boxes: list[BoundingBox]):
        self._boxes = boxes
    def __len__(self):  # length used by MultiSampler
        return len(self._boxes)
    def __iter__(self):
        return iter(self._boxes)

class DummyIteratorSampler(GeoSampler):
    """Sampler whose __iter__ returns a custom iterator object to exercise set semantics."""
    def __init__(self, boxes: list[BoundingBox]):
        self._boxes = boxes
    def __len__(self):
        return len(self._boxes)
    def __iter__(self):
        return DummyIter(self._boxes)

class DummyIter:
    def __init__(self, boxes):
        self._boxes = boxes
        self._idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._idx >= len(self._boxes):
            raise StopIteration
        b = self._boxes[self._idx]
        self._idx += 1
        return b

@pytest.fixture
def sample_boxes():
    return [BoundingBox(minx=i, maxx=i+1, miny=i, maxy=i+1, mint=0, maxt=1) for i in range(5)]

# ---- Tests for MultiSampler ----

def test_multi_sampler_length(sample_boxes):
    s1 = DummySampler(sample_boxes[:2])
    s2 = DummySampler(sample_boxes[2:5])
    ms = MultiSampler([s1, s2])
    assert len(ms) == 2 + 3


def test_multi_sampler_empty():
    ms = MultiSampler([])
    assert len(ms) == 0


def test_multi_sampler_mixed_types(sample_boxes):
    s1 = DummySampler(sample_boxes[:3])
    s2 = DummyIteratorSampler(sample_boxes[3:5])
    ms = MultiSampler((s1, s2))  # tuple iterable
    assert len(ms) == 3 + 2

# ---- Tests for SequentialMultiSampler ----

def test_sequential_multi_sampler_order(sample_boxes):
    s1 = DummySampler(sample_boxes[:2])
    s2 = DummySampler(sample_boxes[2:5])
    seq = SequentialMultiSampler([s1, s2])
    yielded = list(iter(seq))
    assert yielded == sample_boxes  # maintains concatenation order


def test_sequential_multi_sampler_no_samplers():
    seq = SequentialMultiSampler([])
    assert list(iter(seq)) == []

# ---- Tests for RandomMultiSampler ----

def test_random_multi_sampler_exhausts_all(sample_boxes):
    s1 = DummySampler(sample_boxes[:2])
    s2 = DummySampler(sample_boxes[2:5])
    rand = RandomMultiSampler([s1, s2])
    yielded = list(iter(rand))
    # Permutation of all 5 boxes
    assert sorted(b.minx for b in yielded) == [0,1,2,3,4]
    assert len(yielded) == 5


def test_random_multi_sampler_single_sampler(sample_boxes):
    s1 = DummySampler(sample_boxes[:3])
    rand = RandomMultiSampler([s1])
    yielded = list(iter(rand))
    assert len(yielded) == 3


def test_random_multi_sampler_stop_iteration_path():
    # Each sampler yields exactly one element to force StopIteration quickly
    box = BoundingBox(0,1,0,1,0,1)
    samplers = [DummyIteratorSampler([box]) for _ in range(3)]
    rand = RandomMultiSampler(samplers)
    yielded = list(iter(rand))
    assert len(yielded) == 3


def test_random_multi_sampler_deterministic_patch(sample_boxes, monkeypatch):
    s1 = DummySampler(sample_boxes[:2])
    s2 = DummySampler(sample_boxes[2:4])
    s3 = DummySampler(sample_boxes[4:5])
    rand = RandomMultiSampler([s1, s2, s3])
    # Capture created iterators from internal set by first converting to list
    iter_list = [iter(s) for s in [s1, s2, s3]]
    # Monkeypatch random.choice to cycle deterministically
    sequence = iter(iter_list)
    def fake_choice(seq):
        try:
            return next(sequence)
        except StopIteration:
            # fall back to real choice for any remaining
            return real_choice(list(seq))
    monkeypatch.setattr('random.choice', fake_choice)
    yielded = list(iter(rand))
    assert len(yielded) == 5

# ---- Edge cases & robustness ----

def test_random_multi_sampler_no_samplers():
    rand = RandomMultiSampler([])
    assert list(iter(rand)) == []


def test_random_multi_sampler_iterators_removed(sample_boxes):
    # Use iterator sampler so removal branch executed
    samplers = [DummyIteratorSampler(sample_boxes[:1]), DummyIteratorSampler(sample_boxes[1:2])]
    rand = RandomMultiSampler(samplers)
    yielded = list(iter(rand))
    assert len(yielded) == 2


def test_sequential_multi_sampler_iterator_sampler(sample_boxes):
    s1 = DummyIteratorSampler(sample_boxes[:2])
    s2 = DummyIteratorSampler(sample_boxes[2:3])
    seq = SequentialMultiSampler([s1, s2])
    yielded = list(iter(seq))
    assert len(yielded) == 3

# ---- Stress small tokens (BoundingBox default) ----

def test_dummy_sampler_len_consistency():
    b = BoundingBox(0,1,0,1,0,1)
    s = DummySampler([b, b, b])
    assert len(s) == 3
    ms = MultiSampler([s])
    assert len(ms) == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
