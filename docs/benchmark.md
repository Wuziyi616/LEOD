# Benchmark

Please make sure you have downloaded the pre-trained weights following instructions in [install.md](./install.md).

## Evaluation

To evaluate a pre-trained model, set the `dataset`, `dataset.path`, and `checkpoint` fields accordingly.

```Bash
# Gen1 (1649it in total, 470 full sequences)
python val.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ checkpoint="pretrained/Sec.4.2-WSOD_SSOD/gen1-WSOD/rvt-s-gen1x0.01_ss-final.ckpt" use_test_set=1 hardware.gpus=0 hardware.num_workers.eval=8 +experiment/gen1="small.yaml" batch_size.eval=16 model.postprocess.confidence_threshold=0.001 reverse=False tta.enable=False

# Gen4 (1Mpx) (3840it in total, 120 full sequences)
python val.py model=rnndet dataset=gen4 dataset.path=./datasets/gen4/ checkpoint="pretrained/Sec.4.2-WSOD_SSOD/gen4-WSOD/rvt-s-gen4x0.01_ss-final.ckpt" use_test_set=1 hardware.gpus=0 hardware.num_workers.eval=8 +experiment/gen4="small.yaml" batch_size.eval=8 model.postprocess.confidence_threshold=0.001 reverse=False tta.enable=False
```

We also support the `reverse` flag which tests the model on event sequences in the reverse-time order, and `tta.enable` which enables Test-Time Augmentation (model prediction ensembled over horizontal and temporal flip).

### Visualize the detection results

You can apply RVT to the entire event sequence and get continuous detection results as videos:

```Bash
python vis_pred.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ checkpoint=xxx.ckpt +experiment/gen1="small.yaml" model.postprocess.confidence_threshold=0.1 num_video=5 reverse=False
```

The mp4 files will be saved to `./vis/gen1_rnndet_small/pred/`.

## Training

### Pre-training on the limited annotated data

We follow RVT for most of the settings, e.g., batch size, learning rate.
The biggest change we made is early stopping to prevent overfitting.

```Bash
# Gen1 （1 GPU, batch_size=8 per GPU)
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1="small.yaml" training.max_steps=200000

# Gen4 (1Mpx) (2 GPUs, batch_size=12 per GPU)
python train.py model=rnndet hardware.gpus=[0,1] dataset=gen4x0.01_ss +experiment/gen4="small.yaml" training.max_steps=200000
```

Please refer to Appendix A.2 in the paper for different training steps we use under different ratio of data.
In general, we use 200k steps for 1% of the data, 300k steps for 2% of the data, and 400k steps for 5% and more data (because RVT trains for 400k steps on full data).

It is possible to get better results in this pre-training stage, as we did not tune the hyper-parameters very carefully for this stage.

### Generate pseudo dataset for self-training

After obtaining a trained model (either trained on annotated data only, or after first-/second-round self-training), we can use it to generate pseudo labels on all unlabeled events for the next round of model training.

The logic is simple, we run the model on the entire event sequence, and save the detection results in the same format as the original dataset.
This way, we can use the same training pipeline to train the model on the pseudo dataset.
Note that labels are saved as `.npy` files, while the events are soft-linked instead of copied to save storage.
The entire eval/test sets are also soft-linked.

```Bash
# Gen1 (11376it, ~6h predict + ~30min saving on a T4 GPU)
python predict.py model=pseudo_labeler dataset=gen1 dataset.path=./datasets/gen1/ checkpoint="pretrained/Sec.4.2-WSOD_SSOD/gen1-WSOD/rvt-s-gen1x0.01_ss.ckpt" hardware.gpus=0 +experiment/gen1="small.yaml" model.postprocess.confidence_threshold=0.01 tta.enable=True save_dir=./datasets/pseudo_gen1/gen1x0.01_ss/train

# Gen4 (27044it, ~9.5h predict + ~15min saving on a T4 GPU)
python predict.py model=pseudo_labeler dataset=gen4 dataset.path=./datasets/gen4/ checkpoint="pretrained/Sec.4.2-WSOD_SSOD/gen4-WSOD/rvt-s-gen4x0.01_ss.ckpt" hardware.gpus=0 +experiment/gen4="small.yaml" model.postprocess.confidence_threshold=0.01 tta.enable=True save_dir=./datasets/pseudo_gen4/gen4x0.01_ss/train
```

This script also run tests to ensure that the saved labels are in the correct format.
E.g., the saved labels should be the same on frames that are originally annotated.

### Evaluate the quality of the pseudo labels

You may want to inspect the quality of the generated pseudo labels, e.g., their AP and AR.
We provide scripts for doing this:

```Bash
# Gen1 (2839 iter)
python val_dst.py model=pseudo_labeler dataset=gen1x0.01_ss dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss checkpoint=1 +experiment/gen1="small.yaml" model.pseudo_label.obj_thresh=0.01 model.pseudo_label.cls_thresh=0.01

# Gen4 (6764 iter)
python val_dst.py model=pseudo_labeler dataset=gen4x0.01_ss dataset.path=./datasets/pseudo_gen4/gen4x0.01_ss checkpoint=1 +experiment/gen4="small.yaml" model.pseudo_label.obj_thresh=0.01 model.pseudo_label.cls_thresh=0.01
```

### Self-training on pseudo labels

Since our pseudo labels are saved in the same format as real labels, the training commands are almost the same:

```Bash
# Gen1 （1 GPU, batch_size=8 per GPU)
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss +experiment/gen1="small.yaml" training.max_steps=150000 training.learning_rate=0.0005 dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss dataset.ratio=-1 dataset.train_ratio=-1

# Gen4 (1Mpx) (2 GPUs, batch_size=12 per GPU)
python train.py model=rnndet hardware.gpus=[0,1] dataset=gen4x0.01_ss +experiment/gen4="small.yaml" training.max_steps=150000 training.learning_rate=0.0005 dataset.path=./datasets/pseudo_gen4/gen4x0.01_ss dataset.ratio=-1 dataset.train_ratio=-1
```

Note that now we only train for 150k steps and can use a larger learning rate.
This is because we have much denser annotations, which increases the effective batch size, and leads to faster convergence.
We also set `ratio` (WSOD) and `train_ratio` (SSOD) to -1 because now we generate dense pseudo labels on all frames, we do not sub-sample data anymore.
