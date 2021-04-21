### CSC413 final project

David Goldstein 1003477230

Tianxu An 1003099393

Runtian Wang 1003454102


Pretrained models are available for both the old TensorFlow DGCNN and the new simplified Pytorch DGCNN trained using permutation invariant cross entropy loss

The TensorFlow trained model is available at dgcnn/part_seg/train_results/trained_models

The Pytorch model is available at dgcnn/pytorch/checkpoints/exp/models

### Evaluation

To evaluate the SAPIEN data using the TensorFlow model, go to dgcnn/part_seg and run 

```
python3 test-Sapien.py
```

To evaluate the SAPIEN data using the Pytorch model, go to dgcnn/pytorch and run:

```
python3 test-sapien.py
```

### Dataset for Training

Download the data for part segmentation by going to dgcnn/part_seg and running:

```
sh +x download_data.sh
```

### Train

To train, use at least 2 GPUs

Go to dgcnn/pytorch and run:

```
python3 main-perm-simple.py
```
Use the --epochs command to change the number of epochs, for example 

```
python3 main-perm-simple.py --epochs 200
```
