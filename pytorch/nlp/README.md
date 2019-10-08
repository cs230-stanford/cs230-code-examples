# Named Entity Recognition with PyTorch

_Authors: Surag Nair, Guillaume Genthial and Olivier Moindrot_

Take the time to read the [tutorials](https://cs230-stanford.github.io/project-starter-code.html).

Note : all scripts must be run in `pytorch/nlp`.

## Requirements

We recommend using python3 and a virtual env. See instructions [here](https://cs230-stanford.github.io/project-starter-code.html).

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given a sentence, give a tag to each word ([Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition))

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```

## [optional] Download the Kaggle dataset (~5 min)

We provide a small subset of the kaggle dataset (30 sentences) for testing in `data/small` but you are encouraged to download the original version on the [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data) website.

1. **Download the dataset** `ner_dataset.csv` on [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data) and save it under the `nlp/data/kaggle` directory. Make sure you download the simple version `ner_dataset.csv` and NOT the full version `ner.csv`.

2. **Build the dataset** Run the following script

```
python build_kaggle_dataset.py
```

It will extract the sentences and labels from the dataset, split it into train/val/test and save it in a convenient format for our model.

_Debug_ If you get some errors, check that you downloaded the right file and saved it in the right directory. If you have issues with encoding, try running the script with python 2.7.

3. In the next section, change `data/small` by `data/kaggle`

## Quickstart (~10 min)

1. **Build** vocabularies and parameters for your dataset by running

```
python build_vocab.py --data_dir data/small
```

It will write vocabulary files `words.txt` and `tags.txt` containing the words and tags in the dataset. It will also save a `dataset_params.json` with some extra information.

2. **Your first experiment** We created a `base_model` directory for you under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like

```json
{
  "learning_rate": 1e-3,
  "batch_size": 5,
  "num_epochs": 2
}
```

For every new experiment, you will need to create a new directory under `experiments` with a `params.json` file.

3. **Train** your experiment. Simply run

```
python train.py --data_dir data/small --model_dir experiments/base_model
```

It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the development set.

4. **Your first hyperparameters search** We created a new directory `learning_rate` in `experiments` for you. Now, run

```
python search_hyperparams.py --data_dir data/small --parent_dir experiments/learning_rate
```

It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

5. **Display the results** of the hyperparameters search in a nice format

```
python synthesize_results.py --parent_dir experiments/learning_rate
```

6. **Evaluation on the test set** Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set. Run

```
python evaluate.py --data_dir data/small --model_dir experiments/base_model
```

## Guidelines for more advanced use

We recommend reading through `train.py` to get a high-level overview of the training loop steps:

- loading the hyperparameters for the experiment (the `params.json`)
- loading the training and validation data
- creating the model, loss_fn and metrics
- training the model for a given number of epochs by calling `train_and_evaluate(...)`

You can then go through `model/data_loader.py` to understand the following steps:

- loading the vocabularies from the `words.txt` and `tags.txt` files
- creating the sentences/labels datasets from the text files
- how the vocabulary is used to map tokens to their indices
- how the `data_iterator` creates a batch of data and labels and pads sentences

Once you get the high-level idea, depending on your dataset, you might want to modify

- `model/model.py` to change the neural network, loss function and metrics
- `model/data_loader.py` to suit the data loader to your specific needs
- `train.py` for changing the optimizer
- `train.py` and `evaluate.py` for some changes in the model or input require changes here

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/1.2.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)
