import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels


        #self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        n=preds_sub.get_shape().as_list()[0]
        logits=tf.reshape(preds_sub,[n,n,10])
        adj_label=tf.reshape(labels_sub,[n,n,10])
        self.cost=-tf.reduce_mean(tf.multiply(adj_label,tf.log(tf.clip_by_value(logits,1e-5,1))))*(n**2)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        # preds_sub=tf.round(preds_sub)  # by fenzhihui

        # self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        n = preds_sub.get_shape().as_list()[0]
        logits = tf.reshape(preds_sub, [n, n, 10])
        adj_label = tf.reshape(labels_sub, [n, n, 10])
        self.cost = -tf.reduce_mean(tf.multiply(adj_label, tf.log(tf.clip_by_value(logits, 1e-5, 1)))) * (n ** 2)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer


        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        mask = tf.reduce_sum(labels_sub, axis=2)

        preds_sub=tf.argmax(preds_sub,axis=2)
        labels_sub=tf.argmax(labels_sub,axis=2)
        total_labels=tf.reduce_sum(mask)


        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
        #                                    tf.ast(labels_sub, tf.int32))

        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(preds_sub,labels_sub), tf.float32)*mask)/tf.cast(total_labels,tf.float32)
