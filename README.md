# Causal-Flow-Matching on MorphoMNIST
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
