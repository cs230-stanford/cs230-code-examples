# TODO


- enable hyperparameter search
  - `hyperparam_search.py` calling `train.py` over multiple `params.json` in `experiments`
  - `run("train.py --model_dir experiments/exp024")`
  
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

- split the graph into train and eval
  - reuse the weights
  - or use a placeholder??

- if we train again the model, training should start where it stopped
  - add an optional argument for this?

## Tutorial

- explain how to use logger
- explain where to define the model or change it
- explain how to change hyperparameters
- how to feed data...
