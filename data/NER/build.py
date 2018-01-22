"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter
import json


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1)
parser.add_argument('--min_count_tag', default=1)


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab('train/sentences.txt', words)
    size_test_sentences = update_vocab('test/sentences.txt', words)
    print("- done.")

    # Build tag vocab with train and test datasets
    print("Building tag vocabulary...")
    tags = Counter()
    size_train_tags = update_vocab('train/labels.txt', tags)
    size_test_tags = update_vocab('test/labels.txt', tags)
    print("- done.")

    # Assert same number of examples in datasets
    assert size_train_sentences == size_train_tags
    assert size_test_sentences == size_test_tags

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= args.min_count_word]
    tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, 'words.txt')
    save_vocab_to_txt_file(tags, 'tags.txt')
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'number_of_tags': len(tags),
        'pad_word': '<pad>',
        'pad_tag': 'O',
    }
    save_dict_to_json(sizes, 'dataset_params.json')

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
