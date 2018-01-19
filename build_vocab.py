"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter

import numpy as np
import tensorflow as tf

from input_data import load_dataset_from_text
from input_data import input_fn
from model.utils import save_vocab_to_txt_file
from model.utils import save_dict_to_json


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1)
parser.add_argument('--min_count_tag', default=1)


def update_vocabs(inputs, vocab_word, vocab_tag):
    """Update word and tag vocabulary from dataset

    Args:
        inputs: (dict) containing tf.data elements
        vocab_word: (set or Counter) countaining words
        vocab_tag: (set or Counter) countaining tags

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    # Helper function for encoding
    def enc(token):
        """Decodes bytes to utf-8"""
        return token.decode('utf-8')

    # Initialize 
    with tf.Session() as sess:
        # Initialize the iterator over the datasets
        sess.run(inputs['iterator_init_op'])
        dataset_size = 0

        while True:
              try:
                # Read the next batch
                sentence_eval, tags_eval = sess.run([inputs["sentence"], inputs["tags"]])

                # Update the counter of tokens for words and tags
                vocab_word.update([enc(w) for sentence in sentence_eval for w in sentence])
                vocab_tag.update([enc(t) for tags in tags_eval for t in tags])

                # Update the dataset_size
                dataset_size += sentence_eval.shape[0]

              except tf.errors.OutOfRangeError:
                break

    return dataset_size


if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize empty counters for tag and word vocab
    vocab_word, vocab_tag = Counter(), Counter()

    # Create the input data pipeline
    train_sentences = load_dataset_from_text('data/NER/train/sentences.txt')
    train_tags = load_dataset_from_text('data/NER/train/tags.txt')
    train_inputs = input_fn(False, train_sentences, train_tags)

    test_sentences = load_dataset_from_text('data/NER/test/sentences.txt')
    test_tags = load_dataset_from_text('data/NER/test/tags.txt')
    test_inputs = input_fn(False, test_sentences, test_tags)

    # Update vocabularies with train and test datasets
    train_size = update_vocabs(train_inputs, vocab_word, vocab_tag)
    test_size = update_vocabs(test_inputs, vocab_word, vocab_tag)

    # Only keep most frequent tokens
    vocab_word = [tok for tok, count in vocab_word.items() if count >= args.min_count_word]
    vocab_tag = [tok for tok, count in vocab_tag.items() if count >= args.min_count_tag]

    # Save vocabularies to file
    save_vocab_to_txt_file(vocab_word, 'data/NER/vocab_words.txt')
    save_vocab_to_txt_file(vocab_tag, 'data/NER/vocab_tags.txt')

    # Save datasets sizes in json file
    sizes = {
        "train_size": train_size,
        "test_size": test_size
    }
    save_dict_to_json(sizes, 'data/NER/dataset_sizes.json')

        