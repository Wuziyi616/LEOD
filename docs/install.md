# Install

Most of this section is the same as [RVT](https://github.com/uzh-rpg/RVT).
We also heavily rely on a personal package [nerv](https://github.com/Wuziyi616/nerv) for utility functions.

## Environment Setup

Please use [Anaconda](https://www.anaconda.com/) for package management.
```Bash
conda create -y -n leod python=3.9
conda activate leod

conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

python -m pip install tqdm numba hdf5plugin h5py==3.8.0 \
  pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 \
  pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10 \
  opencv-python hydra-core==1.3.2 einops==0.6.0 \
  pytorch-lightning==1.8.6 wandb==0.14.0 torchdata==0.6.0

conda install -y blosc-hdf5-plugin -c conda-forge

# install nerv: https://github.com/Wuziyi616/nerv
git clone git@github.com:Wuziyi616/nerv.git
cd nerv
git checkout v0.4.0  # tested with v0.4.0 release
pip install -e .
cd ..  # go back to the root directory of the project

# (Optional) compile Detectron2 for faster evaluation
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

We also provide a `environment.yml` file which lists all the packages in our final environment.

I sometimes encounter a weird bug where `Detectron2` cannot run on types of GPUs different from the one I compile it on (e.g., if I compile it on RTX6000 GPUs, I cannot use it on A40 GPUs).
To avoid this issue, go to [coco_eval.py](../utils/evaluation/prophesee/metrics/coco_eval.py#L17) and set the `compile_gpu` to the GPU you compile it (the program will not import `Detectron2` when detecting a different GPUs in use).

## Dataset

In this project, we use two datasets: Gen1 and 1Mpx.
Following the convention of RVT, we name Gen1 as `gen1` and 1Mpx as `gen4` (because of the camera used to capture them).
Please download the pre-processed datasets from RVT:

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">1 Mpx</th>
<th valign="bottom">Gen1</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c5ec7c38</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
</tr>
</tbody></table>

After downloading and unzipping the datasets, soft link Gen1 to `./datasets/gen1` and 1Mpx to `./datasets/gen4`.

### Data Splits

To simulate the weakly-/semi-supervised learning settings, we need to sub-sample labels from the original dataset.
An important thing is that we need to keep the data split the same across experiments.
- For semi-supervised setting where we keep the labels for some sequences while making other sequences completely unlabeled, it is relatively easy.
  We just sort the name of event sequences so that their order will be deterministic across runs, and select unlabeled sequences from it.
- For weakly-supervised setting where we sub-sample the labels for all sequences, it is a bit tricky because there are two mode of data sampling in the codebase, and they pre-process events in different ways.
  To have a consistent data split, we create a split file for each setting, which are stored [here](../data/genx_utils/splits/).
  If you want to explore new experimental settings, remember to create your own split files and read from them [here](../data/genx_utils/dataset_streaming.py#L62).

All results in the paper are averaged over three different splits (we offset the index when sub-sampling the data).
Overall, the performance variations are very small across different splits.
Therefore, we only release the split files, config files, and pre-trained weights for the first variant we experimented with.

## Pre-trained Weights

We provide checkpoints for all the models used to produce the final performance in the paper.
In addition, we provide models pre-trained on the limited annotated data (the `Supervised Baseline` method in the paper) to ease your experiments.

Please download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1xBzFovvNbrtBt0YwYcvvrjbV8ozAdCUK/view?usp=sharing) and unzip them to `./pretrained/`.
The weights are grouped by the Section they are presented in the paper.
They naming follows the pattern `rvt-{$MODEL_SIZE}-{$DATASET}x{$RATIO_OF_DATA}_{$SETTING}.ckpt`.

For example, `rvt-s-gen1x0.02_ss.ckpt` is the RVT-S pre-trained on 2% of Gen1 data under the weakly-supervised setting.
`rvt-s-gen4x0.05_ss-final.ckpt` is the RVT-S trained on 5% of 1Mpx data under the semi-supervised setting, and `-final` means it is the LEOD self-trained model (used to produce the results in the paper).

**Note:** it might be a bit confusing, but `ss` means weakly-supervised (all event sequences are sparsely labeled) and `seq` means semi-supervised (some event sequences are densely labeled, while others are completely unlabeled).
