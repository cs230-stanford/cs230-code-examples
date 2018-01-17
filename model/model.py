"""Define the model.
"""

import tensorflow as tf


def model_fn(is_training, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    images = inputs['images']
    labels = inputs['labels']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        if params.model_version == '2_fc':
            h1 = tf.layers.dense(images, 64, activation=tf.nn.relu)
            h1 = tf.layers.dropout(h1, rate=params.dropout_rate, training=is_training)
            logits = tf.layers.dense(h1, 10)
        elif params.model_version == '2_conv_1_fc':
            out = tf.reshape(images, [-1, 28, 28, 1])
            out = tf.layers.conv2d(out, 32, 5, padding='same', activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2)
            out = tf.layers.conv2d(out, 64, 5, padding='same', activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2)
            out = tf.reshape(out, [-1, 7 * 7 * 64])
            out = tf.layers.dense(out, 128)
            out = tf.layers.dropout(out, rate=params.dropout_rate, training=is_training)
            logits = tf.layers.dense(out, 10)
        else:
            raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # Define training metrics
    predictions = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define loss and train_op
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics
    with tf.variable_scope("eval_metrics"):
        eval_metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in eval_metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    local_metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="eval_metrics")
    local_metrics_init_op = tf.variables_initializer(local_metric_variables)

    # Summaries for training
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['loss'] = loss
    model_spec['train_op'] = train_op
    model_spec['local_metrics_init_op'] = local_metrics_init_op
    model_spec['eval_metrics'] = eval_metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    return model_spec
