import numpy as np
import tensorflow as tf
from time import time
import math


from include.data import get_data_set
from include.model import Model

model = Model()
train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model.getModel()
global_accuracy = 0


# PARAMS
_BATCH_SIZE = 256
_EPOCH = 60
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"


# LOSS AND OPTIMIZER
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output[-1], labels=y[-1])
new_loss = -1*(loss2/loss)
model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step,)
model.senoptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(new_loss, global_step=global_step,)
# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

model.var_list = tf.trainable_variables()
model.allvars = tf.trainable_variables()



model.freezeAllExcept("Flaten_and_dense","last")

text_file=None



def train(epoch,xi):
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0

    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]


        batch_xs_List = np.ndarray.tolist(batch_xs)
        train_x_List = np.ndarray.tolist(train_x[xi])
        batch_xs_List.append(train_x_List)
        batch_xs = np.array(batch_xs_List)

        batch_ys_List = np.ndarray.tolist(batch_ys)
        train_y_List = np.ndarray.tolist(train_y[xi])
        batch_ys_List.append(train_y_List)
        batch_ys = np.array(batch_ys_List)

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, model.senoptimizer, new_loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: model.lr(epoch)})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            text_file.write(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))

    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x
                  ):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: model.lr(epoch)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}). Global max accuracy is {:.2f}%"
    text_file.write(mes.format((epoch+1), acc, correct_numbers, len(test_x), global_accuracy))

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        text_file.write(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

        text_file.write("###########################################################################################################")


def main():
    for xi in range(20):
        text_file = open("xi.txt", "w")
        loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output[xi], labels=y[xi])
        new_loss = -1 * (loss2 / loss)
        model.senoptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                    beta1=0.9,
                                                    beta2=0.999,
                                                    epsilon=1e-08).minimize(new_loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        for i in range(_EPOCH):
            text_file.write("\nEpoch: {0}/{1}\n".format((i+1), _EPOCH))
            train(i,xi)
    text_file.close()



if __name__ == "__main__":
    main()


sess.close()
