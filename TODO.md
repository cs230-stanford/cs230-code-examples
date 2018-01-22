# TODO

- change name `eval_metrics` to `metrics`
  - introduces confusion: only have one set of metrics that are said to be an average on the dataset
  - what we log with tqdm is not metrics but "training monitoring"

- move summaries into folders

- enable hyperparameter search
  - `hyperparam_search.py` calling `train.py` over multiple `params.json` in `experiments`
  - `run("train.py --model_dir experiments/exp024")`
  - check that there is no problem with the virtual env and that we don't need to add a preamble
  
- saving
  - if we train again the model, training should start where it stopped
    - add an optional argument for this?

- split the graph into train and eval
  - clean-up
  - reuse the weights with 2 graphs and 2 inputs
  - (or use a placeholder??)

- split `hyperparams_search.py` into two files?
  - one for hyperparam search
  - one for "syntethizing results"


- explore file `explore.py` ?
  - would run the model on some examples (dataset)
  - give access to some errors
  - interaction? (ipython like test?)


- explicitely have a train / dev / test split
  - for the SIGNS dataset, there is only train / test --> do the split in train.py
  - some images are duplicated??? Ignore or clean up the dataset?

- add tf.summary.image for training images?

- make sure there is only images in SIGNS (no .DS_Store)

- add script `data/download_data.sh`
  - remove `.gitkeep`?



## Done

- add logging
  - everything logged to terminal also logged to file in model_dir


- saving
  - saves weights into model_dir every ...
  - only keep best model based on validation accuracy?
    - `if val > best_val: best_save_path = saver.save(...)`
  - also save last model saved

- tensorboard
  - add summaries into model, pass them to model_spec

- add random seed to make experiments reproducible



## Tutorial


- structure of the project (files' roles, experiment pipeline)
- how to run the toy examples
- explain how to use logger
- explain where to define the model or change it
- explain how to change hyperparameters
- how to feed data...

- use github release to have multiple version of the code?

- Explain the general idea of training multiple models, trying different structures...
  - make sure that experiments are reproducible
    - for instance, if model.py has incompatible changes (ex: adds batch norm), previous params.json cannot be run again
    - have to update old params.json to match the new change (ex: put `params.use_bn` argument, and add it to all old `params.json`)
  - give good names to the dirs in `experiments`
  - visualize on tensorboard
  - don't spend too much time watching training progress: launch hyperparam search, let it run and get back later (make sure there is no bug first)
