"""Batch sampler that keeps several spectra from each condition together."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class ConditionGroupedBatchSampler(Sampler[list[int]]):
    """Build fixed-size batches from equal-size within-condition groups.

    The sampler receives condition vectors aligned with the *local* dataset
    indices.  Every emitted batch contains complete condition groups, so a
    within-condition diversity loss always has enough samples to evaluate.
    """

    def __init__(
        self,
        condition_vectors: np.ndarray,
        *,
        batch_size: int,
        samples_per_condition: int,
        shuffle: bool,
        random_seed: int,
        drop_last: bool = False,
    ) -> None:
        values = np.asarray(condition_vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("condition_vectors必须为非空二维数组[N,C]。")
        if not np.isfinite(values).all():
            raise ValueError("condition_vectors包含NaN或无穷值。")

        self.batch_size = int(batch_size)
        self.samples_per_condition = int(samples_per_condition)
        self.shuffle = bool(shuffle)
        self.random_seed = int(random_seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        if self.samples_per_condition < 2:
            raise ValueError("samples_per_condition至少为2。")
        if self.batch_size < self.samples_per_condition:
            raise ValueError("batch_size不能小于samples_per_condition。")
        if self.batch_size % self.samples_per_condition != 0:
            raise ValueError(
                "batch_size必须能被samples_per_condition整除，"
                "避免拆散同条件光谱组。"
            )

        grouped: dict[tuple[float, ...], list[int]] = defaultdict(list)
        for index, row in enumerate(values):
            grouped[tuple(float(value) for value in row)].append(index)

        self._groups = tuple(
            np.asarray(indices, dtype=np.int64)
            for _, indices in sorted(grouped.items())
        )
        for indices in self._groups:
            if indices.size < self.samples_per_condition:
                raise ValueError(
                    "每个条件的样本数都必须不少于samples_per_condition。"
                )
            if indices.size % self.samples_per_condition != 0:
                raise ValueError(
                    "每个条件的样本数必须能被samples_per_condition整除；"
                    "否则会产生无法计算组内多样性的残缺组。"
                )

        total_groups = sum(
            int(indices.size // self.samples_per_condition)
            for indices in self._groups
        )
        groups_per_batch = self.batch_size // self.samples_per_condition
        if not self.drop_last and total_groups % groups_per_batch != 0:
            raise ValueError(
                "条件组总数不能组成完整batch；请调整batch_size或"
                "samples_per_condition。"
            )
        self._length = (
            total_groups // groups_per_batch
            if self.drop_last
            else (total_groups + groups_per_batch - 1) // groups_per_batch
        )

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.random_seed + self._epoch)
        self._epoch += 1

        chunks: list[list[int]] = []
        for original_indices in self._groups:
            indices = original_indices.copy()
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, indices.size, self.samples_per_condition):
                chunks.append(
                    indices[start : start + self.samples_per_condition].tolist()
                )

        if self.shuffle:
            generator.shuffle(chunks)

        groups_per_batch = self.batch_size // self.samples_per_condition
        for start in range(0, len(chunks), groups_per_batch):
            selected = chunks[start : start + groups_per_batch]
            if len(selected) < groups_per_batch:
                if self.drop_last:
                    break
                raise RuntimeError("内部错误：出现残缺条件batch。")
            yield [index for chunk in selected for index in chunk]
