# Trying out DVC with Toy Cat vs Dog Example.

> This is a implementation of the [tutorial](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial) from dvc

The aim is to learn versioning of data, model weights, and other files alongs with the code in git using [dvc](dvc.org).

# Prep before doing DVC.

First let's get the data and the repo for training the cat vs dog example.

```shell
git clone https://github.com/iterative/example-versioning.git
cd example-versioning
```

We can download the data using `dvc get`, which is similar to `wget` and can download any file or directory tracked in DVC repository.

```shell
dvc get https://github.com/iterative/dataset-registry \
          tutorials/versioning/data.zip
unzip -q data.zip
rm -f data.zip
```

Now we have the data downloaded as zip file, here for example it's downloading from github, since it's a toy example, but in actual it can be something like S3 bucket or some cloud storage. We than unzip the data and delete (using `rm`) the zip file.

We now have 1000 images for training and 800 images for validation. The data is in the structure:
```
data
├── train
│   ├── dogs
│   │   ├── dog.1.jpg
│   │   ├── ...
│   │   └── dog.500.jpg
│   └── cats
│       ├── cat.1.jpg
│       ├── ...
│       └── cat.500.jpg
└── validation
   ├── dogs
   │   ├── dog.1001.jpg
   │   ├── ...
   │   └── dog.1400.jpg
   └── cats
       ├── cat.1001.jpg
       ├── ...
       └── cat.1400.jpg
```

Now for our model, we will be using a pretrained `VGG16 with Batch Normalization`, and we would be attaching a head of after the adapative pooling which brings the size to `7x7x512`. We flatten them out and pass to our head which is a one layer neural network to predict cat or dog.

Here instead of training the entire model, we would like to freeze the network, and this could be done by mkaing the VGG parameters have `require_grad` as False, but instead we create the feature map of the data by passing it to VGG and storing it as an `.npy` file. Which can be now used to load directly using `numpy` and send to the head for training.

# DVC

We can now capture the current state data using `dvc add data`.

This command is used instead of `git add` on files or directories that are too large to be tracked with Git: usually input datasets, models, some intermediate results, etc. It tells Git to ignore the directory and puts it into the cache (while keeping a file link to it in the workspace, so we can continue working the same way as before). This is achieved by creating a tiny, human-readable .dvc file that serves as a pointer to the cache.

Now, after we train the model, we have set it up to generate the `model.h5` file which is bascially the weights of our head model and also `metrics.csv` which is the metrics file storing the loss, accruacy, class-wise accuracy for each epoch during training/validation.  
The simplest way to capture the current version of the model is to use dvc `add again`.

```shell
python train.py
dvc add model.h5
```
> We manually added the model output here, which isn't ideal. The preferred way of capturing command outputs is with `dvc run`

Now let's commit the current state of the model and the data along with metrics.

```shell
git add data.dvc model.h5.dvc metrics.csv .gitignore
git commit -m "First model, trained with 1000 images"
git tag -a "v1.0" -m "model v1.0, 1000 images"
```

## How DVC works?

As we saw briefly, DVC does not commit the data/directory and model.h5 file with Git. Instead, dvc add stores them in the cache (usually in .dvc/cache) and adds them to .gitignore.  
In this case, we created data.dvc and model.h5.dvc, which contain file hashes that point to cached data. We then git commit these .dvc files.

## Data Update - more Images

Now, assume we got more annotated data, 500 new images of each cat and dog.
```shell
dvc get https://github.com/iterative/dataset-registry \
          tutorials/versioning/new-labels.zip
unzip -q new-labels.zip
rm -f new-labels.zip
```
It got unziped to same data folder, and respective sub folders of `cat` and `dog` as seen earlier, now with effective 2000 images for training and 800 for validation.

## `dvc run`

DVC run is the preferred way to run the python scrip for training where there are files that are the result of running some code. In our example, `train.py` produces binary files (e.g. `train_bottleneck_features.npy`), the model file `model.h5`, and the metrics file `metrics.csv`.

```shell
dvc run -n train -d train.py -d data \
          -o model.h5 -o test_bottleneck_features.npy \
          -o train_bottleneck_features.npy -M metrics.csv \
          python train.py
```
`dvc run` writes a pipeline stage named train (specified using the -n option) in dvc.yaml. It tracks all outputs (-o) the same way as dvc add does. Unlike dvc add, dvc run also tracks dependencies (-d) and the command (python train.py) that was run to produce the result.

```shell
git add .
git commit -m "model, trained with 2000 images"
git tag -a "v2.0" -m "model v2.0, 2000 images"
```

Now, we can run `git add .`, `git commit` and `git tag` to save the train stage and its outputs to the repository.

## Switching between workspace Versions

The DVC command that helps get a specific committed version of data is designed to be similar to git checkout. All we need to do in our case is to additionally run `dvc checkout` to get the right data into the workspace.

```shell
git checkout v1.0
dvc checkout
```
This does a full workspace checkout, which is checkouts the model, data, code and all of it from the v1. DVC optimizes this operation to avoid copying data or model files each time. So `dvc checkout` is quick even if you have large datasets, data files, or models.

Now, let's say we want to keep the code and model as it is, and just checkout the data from version 1.0

```
git checkout v1.0 data.dvc
dvc checkout data.dvc
```

Now if we run `git status` you'll see that data.dvc is modified and currently points to the v1.0 version of the dataset, while code and model files are from the v2.0 tag.


That's it!

Acknowledgements:

[Official DVC Tutorial](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial)

[DVC Git](https://github.com/iterative/example-versioning)


