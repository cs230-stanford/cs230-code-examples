"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys


def load_dataset(path_csv):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    use_python3 = sys.version_info[0] >= 3
    with (open(path_csv, encoding="windows-1252") if use_python3 else open(path_csv)) as f:
        csv_file = csv.reader(f, delimiter=',')
        dataset = []
        words, tags = [], []

        # Each line of the csv corresponds to one word
        for idx, row in enumerate(csv_file):
            if idx == 0: continue
            sentence, word, pos, tag = row
            # If the first column is non empty it means we reached a new sentence
            if len(sentence) != 0:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
            try:
                word, tag = str(word), str(tag)
                words.append(word)
                tags.append(tag)
            except UnicodeDecodeError as e:
                print("An exception was raised, skipping a word: {}".format(e))
                pass

    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words)))
                file_labels.write("{}\n".format(" ".join(tags)))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/kaggle/ner_dataset.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading Kaggle dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7*len(dataset))]
    val_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/kaggle/train')
    save_dataset(val_dataset, 'data/kaggle/val')
    save_dataset(test_dataset, 'data/kaggle/test')