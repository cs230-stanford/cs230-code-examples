"""Define the model.
"""

import tensorflow as tf


# TODO: instead of using mode, add a parameter "is_training"
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
    elif params.model_version == '2_conv_1_fc':
        out = tf.reshape(images, [-1, 28, 28, 1])
        out = tf.layers.conv2d(out, 32, 5, padding='same', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.layers.conv2d(out, 64, 5, padding='same', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.reshape(out, [-1, 7 * 7 * 64])
        logits = tf.layers.dense(out, 10)
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # Define loss and train_op
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Create initialization operations
    variable_init_op = tf.global_variables_initializer()

    # Metrics for training and evaluation
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Create the model specification and return it
    model_spec = inputs
    model_spec['loss'] = loss
    model_spec['train_op'] = train_op
    model_spec['variable_init_op'] = variable_init_op
    local_metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    model_spec['local_metrics_init_op'] = tf.variables_initializer(local_metric_variables)
    model_spec['metrics'] = metrics
    update_metrics = [op for _, op in metrics.values()]
    model_spec['update_metrics'] = tf.group(*update_metrics)

    # TODO: for eval, we need to return eval_ops
    return model_spec
