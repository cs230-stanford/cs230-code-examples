"""Define the model.
"""

import tensorflow as tf


def model(inputs, mode, params):
    """Model function defining the graph operations.

    Args:
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        mode: (string) can be one of 'train', 'eval' and 'predict'
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    images = inputs['images']
    labels = inputs['labels']

    if params.model_version == '2_fc':
        h1 = tf.layers.dense(images, 64, activation=tf.nn.relu)
        logits = tf.layers.dense(h1, 10)
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # Evaluation metrics
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    init_op = tf.global_variables_initializer()
    # Create the model specification and return it
    model_spec = dict()
    model_spec['accuracy'] = accuracy
    model_spec['loss'] = loss
    model_spec['train_op'] = train_op
    model_spec['init_op'] = init_op

    # TODO: for eval, we need to return eval_ops
    return model_spec
