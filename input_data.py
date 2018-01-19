"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


# def input_fn(is_training, images, labels, params):
#     """Input function for MNIST.

#     Args:
#         is_training: (bool) whether to use the train or test pipeline.
#                      At training, we shuffle the data and have multiple epochs
#         images: (np.ndarray) array of images, with shape (num_samples, 784) and type np.float
#         labels: (np.ndarray) array of labels, with shape (num_samples,) and type np.int64
#         params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
#     """
#     num_samples = images.shape[0]
#     assert images.shape[0] == labels.shape[0],\
#            "Mismatch between images {} and labels {}".format(images.shape[0], labels.shape[0])

#     if is_training:
#         buffer_size = num_samples  # whole dataset into the buffer ensures good shuffling
#     else:
#         buffer_size = 1  # no shuffling

#     # Create a Dataset serving batches of images and labels
#     # We don't repeat for multiple epochs because we always train and evaluate for one epoch
#     dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
#         .shuffle(buffer_size=buffer_size)
#         .batch(params.batch_size)
#         .prefetch(1)  # make sure you always have one batch ready to serve
#     )

#     # Create reinitializable iterator from dataset
#     iterator = dataset.make_initializable_iterator()
#     images, labels = iterator.get_next()
#     init_op = iterator.initializer

#     inputs = {'images': images, 'labels': labels, 'iterator_init_op': init_op}
#     return inputs


def load_dataset_from_text(path_txt, path_vocab=None):
    """Create tf.data Instance from txt file
    
    Args:
        path_txt: (string) path containing one example per line
        path_vocab: (string) path to txt file containing one token per line
    """
    # Load txt file containing tokens
    dataset = tf.data.TextLineDataset(path_txt)

    # Convert line into list of tokens
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # Convert list of tokens into tuple (list of tokens, number of tokens)
    dataset = dataset.map(lambda token: (token, tf.size(token)))

    # # Load vocabularies into lookup table string -> int (word or tag -> id)
    # if path_vocab is not None:
    #     vocab = tf.contrib.lookup.index_table_from_file(path_vocab)
    #     tf.tables_initializer().run()
    #     dataset = dataset.map(lambda tokens, size: (table.lookup(tokens), size))
    #     dataset = dataset.map(lambda id, size: (tf.cast(id, tf.int32), size))

    return dataset


def input_fn(is_training, sentences, tags=None, batch_size=5):
    """Input function for NER

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        path_sentences: (string) path to file containing the sentences
        path_tags: (string) path to file containing the tags
        batch_size: (int) number of element in a batch
        path_vocab_word: (string) path to file containing the word vocab. Txt file, one word per line
        path_vocab_tag: (string) path to file containing the tag vocab. Txt file, one tag per line

    """
    # TODO: num_parallel_calls ?
    # TODO: unknown words

    # Zip the sentence and the tags together
    if tags is not None:
        dataset = tf.data.Dataset.zip((sentences, tags))
    else:
        dataset = sentences

    # Load all the dataset in memory for shuffling is training
    buffer_size = num_samples if is_training else 1

    # Create batches and pad the sentences of different length
    if tags is not None:
        padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
                          tf.TensorShape([])),     # size(sentence)
                         (tf.TensorShape([None]),  # tags of unknown size
                          tf.TensorShape([])))     # size(tags)
    else:
        padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
                          tf.TensorShape([])))     # size(sentence)
    if tags is not None:
        padding_values = (('<pad>',     # sentence padded on the right with '<pad>'
                           0),          # size(source) -- unused
                          ('O',         # tags padded on the right with 'O'
                           0))          # size(target) -- unused
    else:
        padding_values = (('<pad>',     # sentence padded on the right with '<pad>'
                           0))          # size(source) -- unused


    dataset = (dataset
        .padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        .shuffle(buffer_size=buffer_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    iterator = dataset.make_initializable_iterator()

    if tags is not None:
        ((sentence, sentence_size), (tags, tags_size)) = iterator.get_next()
        init_op = iterator.initializer

        inputs = {
            'sentence': sentence, 
            'sentence_size': sentence_size,
            'tags': tags, 
            'tags_size': tags_size,
            'iterator_init_op': init_op
        }
    else:
        (sentence, sentence_size) = iterator.get_next()
        init_op = iterator.initializer

        inputs = {
            'sentence': sentence, 
            'sentence_size': sentence_size,
            'iterator_init_op': init_op
        }
    
    return inputs
