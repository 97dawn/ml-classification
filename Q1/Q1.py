"""
CSE353 HW2
Doeun Kim
doeun.kim@stonybrook.edu
"""
import tensorflow as tf
import random

avg_acc_1, avg_acc_2, avg_acc_3 = 0, 0, 0
for i in range(10):
    training_X_data, training_Y_data, whole_data, test_X_data, test_Y_data = [], [], [], [], []

    #accumulate the data into one list
    with open("data1.txt") as file:
        data = file.readlines()
        whole_data = whole_data + data

    with open("data2.txt") as file:
        data = file.readlines()
        whole_data = whole_data + data

    with open("data3.txt") as file:
        data = file.readlines()
        whole_data = whole_data + data

    #shuffle the data
    random.shuffle(whole_data)

    #seperate the data into training data and test data
    for record in whole_data[:int(len(whole_data)*0.8)]:
        feature = []
        record = record.split("\t")
        feature.append(int(record[0]))
        feature.append(int(record[1]))
        feature.append(int(record[2]))
        feature.append(int(record[3]))
        feature.append(int(record[4]))
        training_X_data.append(feature)
        training_Y_data.append([int(record[5])-1])


    num_of_1, num_of_2, num_of_3 = 0, 0, 0
    for record in whole_data[int(len(whole_data)*0.8):]:
        feature = []
        record = record.split("\t")
        feature.append(int(record[0]))
        feature.append(int(record[1]))
        feature.append(int(record[2]))
        feature.append(int(record[3]))
        feature.append(int(record[4]))
        test_X_data.append(feature)
        test_Y_data.append([int(record[5])-1])
        if int(record[5]) == 1:
            num_of_1 += 1
        elif int(record[5]) == 2:
            num_of_2 += 1
        else:
            num_of_3 += 1

    num_of_class = 3
    X = tf.placeholder(tf.float32, shape=[None,5])
    Y = tf.placeholder(tf.int32, shape=[None,1])
    Y_one_hot = tf.one_hot(Y, num_of_class)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, num_of_class])
    W = tf.Variable(tf.random_normal([5, num_of_class]))
    b = tf.Variable(tf.random_normal([num_of_class]))
    logits = tf.matmul(X, W) + b
    hypo = tf.nn.softmax(logits)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    prediction = tf.argmax(hypo, 1)
    real_output = tf.argmax(Y_one_hot, 1)
    correct_prediction = tf.equal(prediction, real_output)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #train
    c=0
    for step in range(5000):
        c, _ = sess.run([cost,train], feed_dict={X:training_X_data , Y:training_Y_data})
    print("Test #",i+1,"\n")
    print("Cost: ", c,"\nW: ", sess.run(W), "\nb: ", sess.run(b))

    pred = sess.run(prediction, feed_dict={X:test_X_data})
    #test
    pred_1, pred_2, pred_3 = 0, 0, 0
    for i in range(len(test_Y_data)):
        if test_Y_data[i] == [0] and pred[i] == 0:
            pred_1 += 1
        elif test_Y_data[i] == [1] and pred[i] == 1:
            pred_2 += 1
        elif test_Y_data[i] == [2] and pred[i] == 2:
            pred_3 += 1
    print("\nAccuracy of class 1: {:03.2f}\nAccuracy of class 2: {:03.2f}\nAccuracy of class 3: {:03.2f}\n".format((pred_1 / num_of_1),(pred_2 / num_of_2),(pred_3 / num_of_3)))
    avg_acc_1 += pred_1 / num_of_1
    avg_acc_2 += pred_2 / num_of_2
    avg_acc_3 += pred_3 / num_of_3
    print("Accuracy: {:03.2f}\n\n".format(sess.run(accuracy, feed_dict={X: test_X_data, Y:test_Y_data})))
print("Average Accuracy of Class 1: {:03.2f}\nAverage Accuracy of Class 2: {:03.2f}\nAverage Accuracy of Class 3: {:03.2f}".format(avg_acc_1/10, avg_acc_2/10, avg_acc_3/10))