from typing import Dict, List
import numpy as np
import polars as pl
import torch

class TrainDataset:
    def __init__(self, dataset: pl.DataFrame, num_items: int, max_seq_len: int):
        self._dataset = dataset
        self._num_items = num_items
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id'][:-1][-self._max_seq_len :]
        positive_sequence = sample['item_id'][1:][-self._max_seq_len :]
        negative_sequence = np.random.randint(1, self._num_items + 1, size=(len(item_sequence),)).tolist()

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'positive.ids': positive_sequence,
            'positive.length': len(positive_sequence),
            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of samples into batched tensors suitable for model input.

    This function processes a list of dictionaries, each containing keys like '{prefix}.ids'
    and '{prefix}.length' (the length of the sequence for that prefix). For each such prefix, it:
        - Concatenates all '{prefix}.ids' lists from the batch into a single flat list.
        - Collects all '{prefix}.length' values into a list.
        - Converts the resulting lists into torch.LongTensor objects.

    Args:
        batch (List[Dict]): List of sample dictionaries. Each sample must contain keys of the form
            '{prefix}.ids' (list of ints) and '{prefix}.length' (int).

    Returns:
        Dict[str, torch.Tensor]: Dictionary with keys '{prefix}.ids' and '{prefix}.length' for each prefix,
            where values are 1D torch.LongTensor objects suitable for model input.
    """
    processed_batch = {}
    for key in batch[0].keys():
        if key.endswith('.ids'):
            prefix = key.split('.')[0]
            assert '{}.length'.format(prefix) in batch[0]

            processed_batch[f'{prefix}.ids'] = []
            processed_batch[f'{prefix}.length'] = []

            for sample in batch:
                processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch
