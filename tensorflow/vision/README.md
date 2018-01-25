# Hand Signs Recognition with Tensorflow

*Author: Olivier Moindrot*

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



## Quickstart (~15 min)

TODO

## Resources

TODO
