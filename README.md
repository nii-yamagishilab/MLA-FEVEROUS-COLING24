# MLA-FEVEROUS-COLING24
This is an implementation of our paper: **Bridging Textual and Tabular Worlds for Fact Verification: A Lightweight, Attention-Based Model**. If you find our code and/or models useful, please cite:


```bash
@inproceedings{Dabbaghi-LREC-COLING-2024,
    title = "Bridging Textual and Tabular Worlds for Fact Verification: A Lightweight, Attention-Based Model",
    author = "Dabbaghi, Shirin and Kruengkrai, Canasai and Yahyapour, Ramin, Yamagishi, Junichi",
    year = "2024",
}
```


## Setup

We recommend to create a new environment for experiments using conda:

```bash
conda create -y -n mla python=3.9
conda activate mla
```

Then, install `mla` from the repository:

```bash
git https://github.com/nii-yamagishilab/MLA-FEVEROUS-COLING24.git
cd MLA-FEVEROUS-COLING24
pip install -r requirements.txt
pip install feverous
pip install einops
pip install wandb
python -c "import wandb; wandb.login()"
```

To ensure that PyTorch is installed and CUDA works properly, run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

We should see:

```bash
1.12.1+cu113
True
```

:warning: We use PyTorch 1.12.1 with CUDA 11.3. You may need another CUDA version suitable for your environment.

For further development or modification, we recommend installing `pre-commit`:
```bash
pre-commit install
```

## Experiments

See [experiments](experiments).

## Pre-trained modesl 

[Pre-trained models part 1] : https://doi.org/10.5281/zenodo.10901784

[Pre-trained models part 2] : https://doi.org/10.5281/zenodo.10902611


## Acknowledgments

This work is supported by the National Institute of Informatics (NII), Japan.

## Licence

BSD 3-Clause License

Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
