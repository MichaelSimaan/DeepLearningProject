import numpy as np
import tensorflow as tf
from time import time
import math


from include.data import get_data_set
from include.model import Model

firstRun = False
model = Model()
train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model.getModel()
global_accuracy = 0


# PARAMS
_BATCH_SIZE = 255
_EPOCH = 60
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"


# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)

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


if not firstRun:
    model.freeze("conv2","layer2")
    model.freeze("conv1", "layer1")
    model.freeze("conv3", "layer3")

text_file=None



def train(epoch, xi,new_loss=None,text_file=None):
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0
    current_loss = new_loss
    opt = model.senoptimizer
    if firstRun:
        current_loss = loss
        opt = model.optimizer
    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        if not firstRun:

            batch_xs_List = np.ndarray.tolist(batch_xs)
            train_x_List = np.ndarray.tolist(train_x[xi])
            batch_xs_List.insert(0,train_x_List)
            batch_xs = np.array(batch_xs_List)

            batch_ys_List = np.ndarray.tolist(batch_ys)
            train_y_List = np.ndarray.tolist(train_y[xi])
            batch_ys_List.insert(0,train_y_List)
            batch_ys = np.array(batch_ys_List)
        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, opt, current_loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: model.lr(epoch)})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))
            text_file.write(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration) +"\n")
    test_and_save(i_global, epoch,text_file)


def test_and_save(_global_step, epoch,text_file):
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
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x), global_accuracy))
    text_file.write(mes.format((epoch+1), acc, correct_numbers, len(test_x), global_accuracy) +"\n")
    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)
        if firstRun:
            saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        text_file.write(mes.format(acc, global_accuracy) + "\n")
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

        print("###########################################################################################################")
        text_file.write("###########################################################################################################" + "\n")

def main():
    if not firstRun:
        for xi in range(len(train_x)):
            text_file = open(str(xi) + ".txt", "w")
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output[xi], labels=y[xi])
            sum_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
            new_loss = -1*(loss2 / sum_loss)
            model.senoptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                        beta1=0.9,
                                                        beta2=0.999,
                                                        epsilon=1e-08).minimize(new_loss, global_step=global_step,var_list=model.var_list)
            model.reset("conv4","layer4",sess)
            sess.run(tf.global_variables_initializer())
            for i in range(_EPOCH):
                text_file.write("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH)+"\n")
                train(i, xi,new_loss,text_file)
        text_file.close()
    else:
        for i in range(_EPOCH):
            text_file = open("firstrun.txt", "w")
            print("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH))
            train(i, None,loss,text_file)
            text_file.close()

if __name__ == "__main__":
    main()


sess.close()
