# Hand Signs Recognition with PyTorch

*Authors: Surag Nair, Olivier Moindrot and Guillaume Genthial*

Take the time to read the [tutorials](https://cs230-stanford.github.io/project-starter-code.html).

Note: all scripts must be run in folder `pytorch/vision`.

## Requirements

We recommend using python3 and a virtual env. See instructions [here](https://cs230-stanford.github.io/project-starter-code.html).

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given an image of a hand doing a sign representing 0, 1, 2, 3, 4 or 5, predict the correct label.


## Download the SIGNS dataset

For the vision example, we will used the SIGNS dataset created for this class. The dataset is hosted on google drive, download it [here][SIGNS].

This will download the SIGNS dataset (~1.1 GB) containing photos of hands signs making numbers between 0 and 5.
Here is the structure of the data:
```
SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

The images are named following `{label}_IMG_{id}.jpg` where the label is in `[0, 5]`.
The training set contains 1,080 images and the test set contains 120 images.

Once the download is complete, move the dataset into `data/SIGNS`.
Run the script `build_dataset.py` which will resize the images to size `(64, 64)`. The new resized dataset will be located by default in `data/64x64_SIGNS`:

```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS
```



## Quickstart (~10 min)

1. __Build the dataset of size 64x64__: make sure you complete this step before training
```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS
```

2. __Your first experiment__ We created a `base_model` directory for you under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. __Train__ your experiment. Simply run
```
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set.

4. __Your first hyperparameters search__ We created a new directory `learning_rate` in `experiments` for you. Now, run
```
python search_hyperparams.py --data_dir data/64x64_SIGNS --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

5. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

6. __Evaluation on the test set__ Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```


## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the training loop steps:
- loading the hyperparameters for the experiment (the `params.json`)
- loading the training and validation data
- creating the model, loss_fn and metrics
- training the model for a given number of epochs by calling `train_and_evaluate(...)`

You can then have a look at `data_loader.py` to understand:
- how jpg images are loaded and transformed to torch Tensors
- how the `data_iterator` creates a batch of data and labels and pads sentences

Once you get the high-level idea, depending on your dataset, you might want to modify
- `model/net.py` to change the neural network, loss function and metrics
- `model/data_loader.py` to suit the data loader to your specific needs
- `train.py` for changing the optimizer
- `train.py` and `evaluate.py` for some changes in the model or input require changes here

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)

[SIGNS]: https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing
