# Causal-Flow-Matching on MorphoMNIST
## Overview
This repository experiementS causal counterfactual on Morphomnist dataset. There are three partS of the training in this repository in order to do counterfactual in `counterfactual.ipynb`. They are flow_model, sup_pgm and aux_pgm where the latter code and idea are adapted from [High Fidelity Image Counterfactuals with Probabilistic Causal Models](https://arxiv.org/pdf/2306.15764) and its corresponding repository [github](https://github.com/biomedia-mira/causal-gen/tree/main). 
### Flow model
I created a [retified flow model](https://openreview.net/pdf?id=XVjTT1nw5z) training objective condition on low dimensional variables for image generation. More specifically, I trained with [Morphomnist dataset](https://github.com/dccastro/Morpho-MNIST) where conditioning on intensity, thickness and digit.
### sup_pgm
This is used for training the relation between parents node which means no image involves in this stage of training. The dataset for this part of training is `train-morpho.csv`, where all the relation between nodes are predefined: 

$$y := f_y(u_y), \qquad u_y \sim \text{MNIST}$$

$$t := f_t(u_t) = 0.5 + u_t, \qquad u_t \sim \text{Gamma}(10,5) $$

$$i := f_i(t,u_i) = 191 \cdot \sigma(0.5u_i + 2t - 5), \qquad u_i \sim \mathcal{N}(0,1) $$

$$\mathbf{x} := f_x(i,t,y,u_x) = \text{Set}(i, y, \text{Set}(t, y, u_x)), \qquad u_x \sim \text{MNIST}$$ 

where y is digit value, t is thickness, i is intensity and x is image
### sup_aux
In order to measure how well our counterfactual image is, we train a simple CNN classification model $q(pa_x|x)$. Once we obtain our counterfactual image, we could use this trained model to predict $pa_x$ and we could compare it with the parents nodes we input for counterfactual. 
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
## Run 
`counterfactual.ipynb` will produce the main results. In order for this to work, there are three checkpoints we need to obtain:
1. we need to train the flow model via running `run_2.sh`.
2. Then running `pgm_train.sh` to train $p(pa_x)$
3. Finally run `aux_train.sh`
Remember to change the `data_dir` path to your own path.  
## Data
For ease of use, we provide the [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST) dataset we used in datasets/morphomnist. 
