import tensorflow as tf

##########################
# Adapted from tkipf/gcn #
##########################

def masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_sigmoid_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    labels = tf.cast(labels, dtype=tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(logits, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_micro_f1(logits, labels, mask):
    """f1 with masking."""
    predicted = tf.round(tf.nn.sigmoid(logits))

    # Use integers to avoid any nasty FP behaviour
    predicted = tf.cast(predicted, dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)
    mask = tf.cast(mask, dtype=tf.int32)

    # expand the mask so that broadcasting works ([nb_nodes, 1])
    mask = tf.expand_dims(mask, -1)

    # Count true positives, true negatives, false positives and false negatives.
    tp = tf.count_nonzero(predicted * labels * mask)
    tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
    fp = tf.count_nonzero(predicted * (labels - 1) * mask)
    fn = tf.count_nonzero((predicted - 1) * labels * mask)

    # Calculate accuracy, precision, recall and F1 score.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    fmeasure = tf.cast(fmeasure, tf.float32)
    return fmeasure


