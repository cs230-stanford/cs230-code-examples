"""Read, split and save the kaggle dataset for our model"""

import csv
import os


def load_dataset(path_csv):
    """Loads dataset into memory from csv file"""
    def to_str(f):
        # Read file in bytes, need to cast to str for python3
        for l in f: yield str(l)

    with open(path_csv, 'rb') as f:
        csv_file = csv.reader(to_str(f), delimiter=',')
        dataset = []
        words, tags = [], []
        # Each line of the csv corresponds to one word
        for idx, row in enumerate(csv_file):
            if idx == 0: continue
            sentence, word, pos, tag = row
            # If the first column is non empty it means we reached a new sentence
            if sentence != "":
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
            words.append(word)
            tags.append(tag)

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

    # Split the dataset into train, dev and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7*len(dataset))]
    dev_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/kaggle/train')
    save_dataset(dev_dataset, 'data/kaggle/dev')
    save_dataset(test_dataset, 'data/kaggle/test')