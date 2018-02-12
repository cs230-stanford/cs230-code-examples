# Hand Signs Recognition with Tensorflow

*Authors: Olivier Moindrot and Guillaume Genthial*

Take the time to read the [tutorials](https://cs230-stanford.github.io).

Note: all scripts must be run in folder `tensorflow/vision`.

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
Run the script `build_dataset.py` which will resize the images to size `(64, 64)`. The new reiszed dataset will be located by default in `data/64x64_SIGNS`:

```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS
```



## Quickstart (~10 min)

1. __Build the dataset of size 64x64__: make sure you complete this step before training
```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS
```

2. __Your first experiment__ We created a `base_model` directory for you under the `experiments` directory. It countains a file `params.json` which sets the parameters for the experiment. It looks like
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
It will instantiate a model and train it on the training set following the parameters specified in `params.json`. It will also evaluate some metrics on the development set.

4. __Your first hyperparameters search__ We created a new directory `learning_rate` in `experiments` for you. Now, run
```
python search_hyperparams.py --data_dir data/64x64_SIGNS --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

5. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

6. __Evaluation on the test set__ Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```


## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the steps:
- loading the hyperparameters for the experiment (the `params.json`)
- getting the filenames / labels 
- creating the input of our model by zipping the filenames and labels together (`input_fn(...)`), reading the images as well as performing batching and shuffling.
- creating the model (=nodes / ops of the `tf.Graph()`) by calling `model_fn(...)`
- training the model for a given number of epochs by calling `train_and_evaluate(...)`


Once you get the high-level idea, depending on your dataset, you might want to modify
- `model/model_fn.py` to change the model
- `model/input_fn.py` to change the way you read data
- `train.py` and `evaluate.py` if somes changes in the model or input require changes here

If you want to compute new metrics for which you can find a [tensorflow implementation](https://www.tensorflow.org/api_docs/python/tf/metrics), you can define it in the `model_fn.py` (add it to the `metrics` dictionnary). It will automatically be updated during the training and will be displayed at the end of each epoch.

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

Introduction to the `tf.data` pipeline
- [programmer's guide](https://www.tensorflow.org/programmers_guide/datasets)
- [reading images](https://www.tensorflow.org/programmers_guide/datasets#decoding_image_data_and_resizing_it)






[SIGNS]: https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing
