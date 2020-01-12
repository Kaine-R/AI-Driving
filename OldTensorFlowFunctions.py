from __future__ import absolute_import, division, print_function, unicode_literals

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame
import gameFunction as gf
from car import Car

import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
import pylab
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    print("TensorFlow version: ", tf.__version__)
    # model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    # model.compile(optimizer='sgd', loss='mean_squared_error')
    #
    # xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 8.0], dtype=float)
    # ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 15.0], dtype=float)
    #
    # model.fit(xs, ys, epochs=500)
    # print(model.predict([20.0]))

    dataSize = 24  # Total Data (includes training and testing)
    trainingPer = .7  # Percent of total data used as training
    trainingSize = int(dataSize * trainingPer)  # Number of training data

    x = np.linspace(-1, 1, dataSize)  # Creating input, x. Where numbers are evenly distributed from -1 to 1.
    xx = np.linspace(0, 0, dataSize)
    print(x)

    # np.random.shuffle(x)
    # print(x)

    y = 2 * x + 2 + np.random.normal(0, 0.2, (dataSize,))  # Creates expected output, y.
    print(y)

    # Puts the data into list that separate training and testing
    xTrain, yTrain = [x[:trainingSize], xx[:trainingSize]], y[:trainingSize]
    xTest, yTest = [x[trainingSize:], xx[trainingSize:]], y[trainingSize:]

    # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorCallback = keras.callbacks.TensorBoard(log_dir=logdir)

    model = keras.models.Sequential([
        keras.layers.Dense(16, input_dim=1),  # activation="relu"
        keras.layers.Dense(1),  # activation="sigmoid" changes things to 0-1 for probability
    ])

    model.compile(
        loss='mse',  # keras.losses.mean_squared_error
        optimizer=keras.optimizers.SGD(lr=0.2),
    )

    print("Training ... With default parameters, this takes less than 10 seconds.")
    training_history = model.fit(
        xTrain,  # input
        yTrain,  # output
        batch_size=trainingSize,
        verbose=0,  # Suppress chatty output; use Tensorboard instead
        epochs=100,
        validation_data=(xTest, yTest),
        # callbacks=[tensorCallback],
    )

    print("Average test loss: ", np.average(training_history.history['loss']))
    print(model.predict([10, 25, 30]))
    print("---------")


def opmain():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)

    # for i in range(5):
    #     plt.grid(False)
    #     plt.imshow(test_images[i], cmap=plt.cm.binary)
    #     plt.xlabel("Actual: " + class_name[test_labels[i]])
    #     plt.title("Prediction: " + class_name[np.argmax(predictions[i])])
    #     plt.show()


def notmain():
    pygame.init()
    screen = pygame.display.set_mode((600, 450))
    pygame.display.set_caption("AI DRIVING")
    clock = pygame.time.Clock()

    ai = Car(screen)

    road1, road2, road3, road4 = gf.setMap()

    BLACK = (250, 250, 250)
    WHITE = (5, 5, 5)
    GRAY = (150, 150, 150)
    RED = (200, 100, 100)
    GREEN = (100, 200, 100)
    BLUE = (100, 100, 200)
    timer = 0

    rect1 = pygame.Rect(0, 0, 20, 20)
    currentRun = 1

    while True:
        screen.fill(GRAY)

        ai.hit = False

        currentRoad, roadNum = gf.selectRoad(ai.getPos(), [road1, road2, road3, road4])
        preload = gf.preloadRoad(ai.getPos(), [road1, road2, road3, road4], roadNum)

        for point in ai.points:
            if pygame.Rect.collidepoint(rect1, point[0], point[1]):
                ai.hit = True

        if currentRoad[2].overlap(ai.mask, (int(ai.x - currentRoad[1].x) - 11, int(ai.y - currentRoad[1].y) - 11)):
            ai.hit = True

        for road in [road1, road2, road3, road4]:
            screen.blit(road[0], road[1])

        if ai.hit:
            pygame.draw.rect(screen, RED, rect1)
            timer += 1
        else:
            pygame.draw.rect(screen, GREEN, rect1)
            if timer > 0:
                timer -= 1
        if timer > 20:
            ai.reset()
            currentRun += 1

        # ----------------------------------
        inputData = [ai.brain.getNodes()]
        result = ai.score

        if currentRun % 10 == 0:
            model = keras.models.Sequential([
                keras.layers.Dense(16, input_dim=1),
                keras.layers.Dense(1),
            ])

            model.compile(
                loss="mse",  # sparse_categorical_crossentropy
                optimizer=keras.optimizers.SGD(lr=0.2)  # adam
            )  # metrics=["accuracy"]

            model.fit(inputData, result, epochs=10)
        # epoch number the time the program sees this info, so they will see inputX 10 times
        # -----------------------------------

        gf.checkEvent(ai)
        ai.update()
        ai.draw()
        ai.drawLines(GREEN)
        ai.drawContact(ai.scan(currentRoad, preload), BLACK)

        pygame.display.flip()
        clock.tick(60)


def testing():
    s = tf.compat.v1.Session()

    # (a1, a2), (s1, s2) = tf.contrib.keras.datasets.mnist.load_data()
    # print(a1.shape[0], end="------------------\n\n")
    # print(a2.shape[0], end="------------------\n\n")
    # high = 0
    # for i in a2:
    #     high = a2 if high > a2 else high

    t = tf.constant([1, 2, 3, 4, 5])
    tf.shape(t)
    tf.rank(t)
    # print(t)
    # print("----")

    # Building graph
    x = tf.compat.v1.placeholder(tf.int32)
    y = tf.compat.v1.placeholder(tf.int32)
    z = x + y

    # Printing Graph
    # print(s.run(z, feed_dict={x: s.run(t), y: [1, 1, 1, 1, 1]}))

    LOGDIR = './graphs'
    tf.compat.v1.reset_default_graph()

    # writer = tf.summary.FileWriter(LOGDIR)
    # writer.add_graph(s.graph)
    # summary_op = tf.summary.merge_all()
    xTrain1 = [*range(500)]
    yTrain1 = []
    for i in range(500):
        even = 1 if i % 2 == 0 else 0
        yTrain1.append(even)
    xTest1, yTest1 = [-1, 0], [-1, 1]

    xTrain, yTrain = tf.constant(xTrain1), tf.constant(yTrain1)
    xTest, yTest = tf.constant([-1, 0]), tf.constant([-1, 1])

    layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])

    model = tf.keras.Sequential([
        layer0
    ])

    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.1))

    trainModel = model.fit(xTrain, yTrain, steps_per_epoch=1, epochs=1000, verbose=False)

    num = []
    winrate = []
    for i in range(200):
        num.append(np.round(np.random.normal(0, 200), 0))
    attempts = np.around(model.predict([num]), 0)
    for j in range(len(attempts)):
        if attempts[j][0] == np.round((num[j]+1) % 2, 0):
            winrate.append(True)
        else:
            winrate.append(False)
        attempts[j] = 0 if attempts[j] < 0 else attempts[j]
        attempts[j] = 1 if attempts[j] > 1 else attempts[j]
        print("Test: {}, Number: {} Result: {} Expected: {} || {}" .format(j, num[j], attempts[j][0], round((num[j]+1) % 2, 0), winrate[j]))
    percent = 0
    for k in winrate:
        if k == True:
            percent += 1
    print(percent/len(num)*100)

    pylab.xlabel("Epoch Number")
    pylab.ylabel("Loss Num")
    pylab.plot(trainModel.history["loss"])
    pylab.show()

    # trainData = tf.compat.v1.estimator.inputs.numpy_input_fn(
    #     {"x": xTrain},
    #     yTrain,
    #     num_epochs=None,
    #     shuffle=True
    # )
    # testData = tf.compat.v1.estimator.inputs.numpy_input_fn(
    #     {"x": xTest},
    #     yTest,
    #     num_epochs=1,
    #     shuffle=False
    # )
    # featureSpec = [tf.feature_column.numeric_column("x", shape=128)]
    #
    # estimator = tf.estimator.LinearClassifier(
    #     featureSpec,
    #     n_classes=2,
    #     # model_dir="./graphs/canned/linear"
    # )

    # estimator.train(trainData, steps=None)

    # evaluation = estimator.evaluate(input_fn=testData)
    # print(evaluation)

    # print(xTrain)
    # pylab.plot(xTrain1, yTrain1, 'b.')
    # pylab.show()

    # ___________________________________________________________________________

    # learningRate = 0.01
    # trainingIteration = 30
    # batchSize = 5
    # displayStep = 2

    # x = tf.placeholder("float", [None, 1])
    # y = tf.placeholder("float", [None, 2])
    #
    # w = tf.Variable(tf.zeros([1, 2]))
    # b = tf.Variable(tf.zeros([2]))

    # with tf.name_scope("WX_b") as scope:
    #     model = tf.nn.softmax(tf.matmul(x, w) + b)
    #
    # # w_h = tf.histogram_summary("weights", w)
    # # b_h = tf.histogram_summary("biases", b)
    #
    # with tf.name_scope("cost") as scope:
    #     costFunction = -tf.reduce_sum(y*tf.log(model))
    #     # tf.scalar_summary("costFunction", costFunction)
    #
    # with tf.name_scope("train") as scope:
    #     optimizer = tf.train(learningRate).minimize(costFunction)
    #
    # init = tf.initialize_all_variables()
    #
    # mergedSummary = tf.merge_all_summaries()

testing()
