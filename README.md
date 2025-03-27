# SRGNN-EMO

Semi-supervised Multi-relational Graph Neural Network for Nuanced Music Emotion Recognition (SRGNN-EMO) utilizes graph-based deep learning methods to predict nuanced emotions evoked through music effectively.

## Features
- Multi-relational graph neural network approach
- Semi-supervised learning for robust predictions
- Compatibility with various music feature representations (e.g., musicnn, jukebox, maest)

## Dataset
The dataset, including preprocessed features, will be published soon. Data preprocessing scripts expect:
- Node features (as `.npy` files)
- Graph information (`graph.pkl` containing edge indices, weights, and types) 
- Labels dictionary

Run ```build_graphs.py``` in ```preprocessing``` to generate the multirelation graphs.

## Requirements
- Python >= 3.8
- PyTorch
- PyTorch Geometric
- faiss
- scikit-learn

## Usage

### Running the Model

You can run the model using:

```bash
python main.py [OPTIONS]
```

### Options
The following command-line arguments can be configured:

- `--dataset`, `-d`: Dataset name (`m4a`)
- `--rep`: Feature representation type (`musicnn`, `jukebox`, `maest`)
- `--input-dim`: Dimension of input features (default: 50)
- `--dim`: Dimension of hidden layers (default: 100)
- `--layers`: Number of GNN layers (default: 1)
- `--gnn-dropout`: Dropout rate for GNN layers (default: 0.1)
- `--dropout`: Dropout rate for regressor layers (default: 0.1)
- `--projection`: Dimension of projection layer; set to 0 to disable
- `--neighbors`: Number of neighbors sampled by NeighborLoader (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--l2`: L2 regularization penalty (default: 1e-5)
- `--clusters`: Number of clusters used in semi-supervised learning (default: 8)
- `--k`: Parameter `k` used in clustering (default: 8)
- `--confidence-threshold`, `-ct`: Confidence threshold for pseudo-labeling (default: 0.1)
- `--tau`, `-t`: Temperature parameter for semi-supervised training (default: 0.1)
- `--alpha`, `-a`, `--beta`, `-b`: Parameters controlling loss function components (default: 0.0)
- `--epochs`, `-e`: Number of training epochs (default: 1000)
- `--batch-size`, `-bs`: Batch size for training (default: 100)
- `--patience`: Early stopping patience (default: 30)
- `--seed`: Random seed (default: 2020)
- `--ratio`: Ratio for train-test split (default: 0.2)
- `--folds`: Number of folds for cross-validation (default: 10)

### Example
```bash
python main.py --dataset m4a --rep jukebox
```

## Citation
Please cite our work if you use this model:

```bibtex
@inproceedings{SRGNN_EMO,
  title={Semi-supervised Multi-relational Graph Neural Network for Nuanced Music Emotion Recognition},
  author={TBA},
  year={TBA},
  publisher={TBA}
}
```

