# Causal-Flow-Matching on MorphoMNIST
## Overview

## Repository Structure

```
causal-retified-flow/
├── checkpoints/
│   └── t_i_d/
│       ├── aux_60k/
│       │   ├── checkpoint.pt
│       │   ├── events.out.tfevents.1754836776.179296162c89.95085.0
│       │   ├── events.out.tfevents.1754836776.179296162c89.95085.1
│       │   └── trainlog.txt
│       ├── flow_matching_exp/
│       │   ├── checkpoint.pt
│       │   ├── events.out.tfevents.1755080344.bd7c337a5d5e.3524.0
│       │   ├── events.out.tfevents.1755080344.bd7c337a5d5e.3524.1
│       │   └── trainlog.txt
│       └── pgm_60k/
│           ├── checkpoint.pt
│           ├── events.out.tfevents.1754846404.179296162c89.143615.0
│           ├── events.out.tfevents.1754846404.179296162c89.143615.1
│           ├── joint_data.pdf
│           └── trainlog.txt
├── datasets/
│   └── morphomnist/
│       ├── args.txt
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte.gz
│       ├── t10k-morpho.csv
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte.gz
│       └── train-morpho.csv
├── src/
│   ├── morphomnist/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── measure.py
│   │   ├── morpho.py
│   │   ├── perturb.py
│   │   ├── skeleton.py
│   │   └── util.py
│   ├── notebook/
│   │   ├── counterfactual.ipynb
│   │   └── eval_flow_matching_exp.txt
│   ├── pgm/
│   │   ├── aux_train.sh
│   │   ├── flow_pgm.py
│   │   ├── layer.py
│   │   ├── pgm_train.sh
│   │   ├── train_pgm.py
│   │   └── utils_pgm.py
│   ├── dataset.py
│   ├── flow_model.py
│   ├── flow_model_2.py
│   ├── hps.py
│   ├── main.py
│   ├── run.sh
│   ├── run_2.sh
│   ├── train.py
│   ├── train_fm_setup.py
│   └── utils.py
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
└── tree.py
```
## Requirements
1. To run the code firstly install python environment:
   ```bash
   conda create -n causal-gen python=3.8
   ```
   ```bash
   conda activate causal-gen
   ```
2. Then install pytorch environments:
   ```bash
   python -m pip install \
   torch==2.0.0+cu117 \
   torchvision==0.15.0+cu117 \
   torchaudio==2.0.0+cu117 \
   --index-url https://download.pytorch.org/whl/cu117 \
   --trusted-host download.pytorch.org
   ```
3. Finally install all other requirements file
   ```bash
   pip install -r requirements.txt
   ```
## Data
For ease of use, we provide the [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST) dataset we used in datasets/morphomnist. 
