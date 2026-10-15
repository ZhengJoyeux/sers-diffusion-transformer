"""Audit a trained D4.2 checkpoint without regenerating or refitting state."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.checkpoint_manager import load_checkpoint_file


def main() -> None:
    parser = argparse.ArgumentParser(description="检查D4.2正式checkpoint结构。")
    parser.add_argument("--checkpoint", required=True)
    arguments = parser.parse_args()
    path = Path(arguments.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    metadata = checkpoint.get("metadata")
    configuration = checkpoint.get("configuration")
    if not isinstance(metadata, dict) or not isinstance(configuration, dict):
        raise RuntimeError("checkpoint缺少metadata或configuration。")
    state = metadata.get("conditional_prior_residual_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint缺少conditional_prior_residual_state。")
    records = state.get("conditions")
    condition_metadata = metadata.get("conditioning_metadata")
    available = (
        condition_metadata.get("conditions")
        if isinstance(condition_metadata, dict)
        else None
    )
    if not isinstance(records, dict) or not isinstance(available, dict):
        raise RuntimeError("checkpoint中的条件记录无效。")
    if set(records) != set(available):
        raise RuntimeError("条件先验集合与14维条件集合不一致。")
    train_indices = metadata.get("training_indices")
    validation_indices = metadata.get("validation_indices")
    test_indices = metadata.get("test_indices")
    if not all(isinstance(value, list) for value in (train_indices, validation_indices, test_indices)):
        raise RuntimeError("checkpoint缺少固定划分索引。")
    per_condition_counts = {
        int(record.get("number_of_training_spectra", -1))
        for record in records.values()
        if isinstance(record, dict)
    }
    if per_condition_counts != {12}:
        raise RuntimeError(
            f"条件先验不是严格train-only 12条拟合：{sorted(per_condition_counts)}"
        )
    lengths = Counter(
        int(record.get("valid_length", 0)) for record in records.values()
    )
    model_state = checkpoint.get("diffusion_state")
    if not isinstance(model_state, dict) or not any(
        "condition_film" in key for key in model_state
    ):
        raise RuntimeError("共享DDPM状态中没有全层条件FiLM参数。")
    conditioning = configuration.get("conditioning", {}) or {}
    if conditioning.get("injection") != "input_and_all_resnet_blocks_film":
        raise RuntimeError("checkpoint未配置全层FiLM条件注入。")

    print("===== D4.2 checkpoint检查通过 =====")
    print(f"checkpoint：{path}")
    print(f"step：{int(checkpoint.get('step', 0))}")
    print(f"共享条件DDPM：1个；条件先验状态：{len(records)}套")
    print(
        f"训练/验证/测试：{len(train_indices)}/"
        f"{len(validation_indices)}/{len(test_indices)}"
    )
    print(
        "条件先验有效点数："
        + "，".join(f"{length}点×{count}条件" for length, count in sorted(lengths.items()))
    )
    print("每套先验只由该条件12条训练光谱拟合：通过")
    print("14维条件＋输入层条件＋所有ResnetBlock FiLM：通过")


if __name__ == "__main__":
    main()
