import json
import random
from typing import List, Any, Tuple


def load_dataset_split(
    path,
    split: str = "train",
    n_samples: int | None = None,
    seed: int | None = None,
    split_seed: int | None = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> List[Any]:
    """
    Load a JSON dataset and return a deterministic split.
    Splits are produced by shuffling indices with a fixed split seed.
    The legacy `seed` argument is retained for backwards compatibility.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")

    n_total = len(data)
    if n_total == 0:
        return []

    if split_seed is None:
        split_seed = 42 if seed is None else seed

    rng = random.Random(split_seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    n_test = n_total - n_train - n_val

    if split == "train":
        split_indices = indices[:n_train]
    elif split == "val":
        split_indices = indices[n_train:n_train + n_val]
    else:
        split_indices = indices[n_train + n_val:n_train + n_val + n_test]

    # Shuffle the isolated split using the run's seed before selecting n_samples
    run_seed = seed if seed is not None else 42
    rng_sample = random.Random(run_seed)
    rng_sample.shuffle(split_indices)

    if n_samples is not None:
        split_indices = split_indices[:n_samples]

    return [data[i] for i in split_indices]
