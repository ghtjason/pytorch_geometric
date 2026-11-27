"""This example trains the Relational Graph Transformer (RelGT) on a RelBench
node-level task.

Please refer to:
1. https://arxiv.org/abs/2505.10960 for RelGT,
2. https://github.com/snap-stanford/relgt for the official implementation,
   and
3. https://arxiv.org/abs/2407.20060 for RelBench.

The default hyperparameters follow the paper's setup for datasets with fewer
than one million training nodes, such as rel-f1, with one layer and a dropout
of 0.3.

Test results on rel-f1 are listed below. The RDL baseline, the defaults
(Table 8) and the best configuration (Table 1) are taken from the paper.
This example reports the mean and standard deviation over five seeds with
`--epochs 40`.

| Task                  | RDL    | Defaults | Best   | This example  |
| --------------------- | ------ | -------- | ------ | ------------- |
| driver-dnf (AUROC)    | 0.7262 | 0.7434   | 0.7587 | 0.712 ± 0.023 |
| driver-top3 (AUROC)   | 0.7554 | 0.7845   | 0.8352 | 0.821 ± 0.008 |
| driver-position (MAE) | 4.0220 | 4.9420   | 3.9170 | 4.604 ± 0.229 |
"""
import argparse
import copy
import math
import operator
import os
from typing import List, Optional

import numpy as np
import torch
from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset, get_dataset_names
from relbench.modeling.graph import (
    get_node_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.nn import HeteroEncoder
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, get_task_names
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.contrib.nn.models import RelGT
from torch_geometric.contrib.transforms import RelGTTokenSampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            'sentence-transformers/average_word_embeddings_glove.6B.300d',
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


def forward(model: RelGT, batch: HeteroData) -> Tensor:
    pred = model(
        batch.tf_dict,
        batch.token_node_idx,
        batch.token_node_type,
        batch.token_hop,
        batch.token_rel_time,
        batch.token_edge_index,
    )
    return pred.view(-1) if pred.size(1) == 1 else pred


def train(
    model: RelGT,
    loader: NeighborLoader,
    entity_table: str,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_steps: int,
) -> float:
    model.train()

    total_loss = total_examples = 0
    total = min(len(loader), max_steps)
    for step, batch in enumerate(tqdm(loader, total=total), start=1):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = forward(model, batch)
        loss = loss_fn(pred, batch[entity_table].y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * pred.size(0)
        total_examples += pred.size(0)
        if step >= max_steps:
            break

    return total_loss / total_examples


@torch.no_grad()
def test(
    model: RelGT,
    loader: NeighborLoader,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    model.eval()

    # Fix the random draws of every evaluation without affecting training:
    devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            preds.append(forward(model, batch).cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='rel-f1',
                        choices=get_dataset_names())
    parser.add_argument('--task', type=str, default='driver-top3')
    parser.add_argument('--num_tokens', type=int, default=300,
                        help='The number of tokens K per seed node')
    parser.add_argument(
        '--num_neighbors', type=int, nargs='+', default=[5000, 1000],
        help='The maximum number of neighbors per hop and edge type, from '
        'which the tokens are drawn')
    parser.add_argument('--temporal_strategy', type=str, default='uniform',
                        choices=['uniform', 'last'])
    parser.add_argument('--channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_centroids', type=int, default=4096)
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help='The normalization of the feed-forward networks')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attn_dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_steps_per_epoch', type=int, default=3000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cache_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.task not in get_task_names(args.dataset):
        raise ValueError(f"Invalid task '{args.task}' for dataset "
                         f"'{args.dataset}' (choose from "
                         f"{get_task_names(args.dataset)})")
    dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)

    # Clamp regression predictions to the 2nd and 98th percentiles of the
    # training targets, as in the official implementation:
    if task.task_type == TaskType.REGRESSION:
        out_channels, tune_metric, higher_is_better = 1, 'mae', False
        loss_fn = torch.nn.L1Loss()
        target = task.get_table('train').df[task.target_col].to_numpy()
        clamp_min, clamp_max = np.percentile(target, [2, 98])

        def postprocess(pred: np.ndarray) -> np.ndarray:
            return np.clip(pred, clamp_min, clamp_max)

    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels, tune_metric, higher_is_better = 1, 'roc_auc', True
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def postprocess(pred: np.ndarray) -> np.ndarray:
            return torch.from_numpy(pred).sigmoid().numpy()

    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
        tune_metric, higher_is_better = 'multilabel_auprc_macro', True
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def postprocess(pred: np.ndarray) -> np.ndarray:
            return torch.from_numpy(pred).sigmoid().numpy()

    else:
        raise ValueError(f"Unsupported task type '{task.task_type}'")

    # Build a temporal heterogeneous graph with `TensorFrame` node features:
    db = dataset.get_db()
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=256,
        ),
        cache_dir=os.path.join(args.cache_dir, args.dataset, 'materialized'),
    )

    # Replace missing values, which cause NaN gradients in the `torch_frame`
    # encoders:
    for node_type in data.node_types:
        feat_dict = data[node_type].tf.feat_dict
        for stype, feat in feat_dict.items():
            if isinstance(feat, Tensor):
                feat_dict[stype] = torch.nan_to_num(feat, nan=0.0, posinf=1e6,
                                                    neginf=-1e6)

    # Draw `num_tokens` tokens per seed row from its temporal 2-hop
    # neighborhood:
    loader_dict = {}
    for split in ['train', 'val', 'test']:
        table_input = get_node_train_table_input(task.get_table(split), task)
        transforms = [RelGTTokenSampler(num_tokens=args.num_tokens)]
        if table_input.transform is not None:  # Attaches the targets.
            transforms.insert(0, table_input.transform)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors={
                edge_type: args.num_neighbors
                for edge_type in data.edge_types
            },
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            time_attr='time',
            temporal_strategy=args.temporal_strategy,
            transform=T.Compose(transforms),
            batch_size=args.batch_size,
            shuffle=split == 'train',
            # Batch normalization requires more than one example per batch:
            drop_last=split == 'train',
            num_workers=args.num_workers,
            # Evaluation needs new workers to reproduce their random state:
            persistent_workers=split == 'train' and args.num_workers > 0,
        )

    feature_encoder = HeteroEncoder(
        channels=args.channels,
        node_to_col_names_dict={
            node_type: data[node_type].tf.col_names_dict
            for node_type in data.node_types
        },
        node_to_col_stats=col_stats_dict,
    )
    model = RelGT(
        channels=args.channels,
        out_channels=out_channels,
        node_types=data.node_types,
        feature_encoder=feature_encoder,
        num_layers=args.num_layers,
        heads=args.heads,
        num_centroids=args.num_centroids,
        max_hops=len(args.num_neighbors),
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        norm=args.norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    better = operator.ge if higher_is_better else operator.le
    best_val = -math.inf if higher_is_better else math.inf
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        loss = train(model, loader_dict['train'], task.entity_table, optimizer,
                     loss_fn, device, args.max_steps_per_epoch)
        val_pred = postprocess(
            test(model, loader_dict['val'], device, args.seed))
        val_metrics = task.evaluate(val_pred, task.get_table('val'))
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ' +
              ', '.join(f'Val {key}: {value:.4f}'
                        for key, value in val_metrics.items()))

        if better(val_metrics[tune_metric], best_val):
            best_val = val_metrics[tune_metric]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_pred = postprocess(
        test(model, loader_dict['test'], device, args.seed))
    test_metrics = task.evaluate(test_pred)
    print(', '.join(f'Test {key}: {value:.4f}'
                    for key, value in test_metrics.items()))


if __name__ == '__main__':
    main()
