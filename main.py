from __future__ import absolute_import, division, print_function, unicode_literals

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame
import gameFunction as gf
from text import text
from car import Car

import copy

import testData

import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
import pylab
import matplotlib.pyplot as plt
import os

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def notmain():
    pygame.init()
    screen = pygame.display.set_mode((600, 450))
    pygame.display.set_caption("AI DRIVING")
    clock = pygame.time.Clock()

    BLACK = (250, 250, 250)
    WHITE = (5, 5, 5)
    GRAY = (150, 150, 150)
    RED = (200, 100, 100)
    GREEN = (100, 200, 100)
    BLUE = (100, 100, 200)
    timer = 0
    endTimer = 0
    currentRun = 1
    history = []

    road1, road2, road3, road4 = gf.setMap()
    gen = text(screen, "Gen: " + str(currentRun), (80, 240))
    ai = [Car(screen), Car(screen), Car(screen), Car(screen), Car(screen)]

    for car in ai:
        for i in range(3):
            car.brain.createRandNode()

    while True:
        screen.fill(GRAY)

        currentRoad, roadNum = gf.selectRoad(ai[0].getPos(), [road1, road2, road3, road4])
        preload = gf.preloadRoad(ai[0].getPos(), [road1, road2, road3, road4], roadNum)

        for car in ai:
            if currentRoad[2].overlap(car.mask,
                                      (int(car.x - currentRoad[1].x) - 11, int(car.y - currentRoad[1].y) - 11)):
                car.hit = True
            gf.checkpointCollision(car)
            car.checkSpeed()

        timer += 1
        timer += gf.checkSpace()
        if timer > 400 + (2 * currentRun):
            endTimer += 1

        if endTimer > 25:
            endTimer = 0

            found = -1
            for num, car in enumerate(ai, 0):
                if car.score != 0:
                    history.append((car.brain.getTensorflowData(), car.score))
                car.brain.randMutate()
                car.reset()

            timer = 0
            currentRun += 1
            gen.prep("Gen: " + str(currentRun))

            if currentRun % 40 == 0 and currentRun != 0:
                xTrain = tf.constant([history[0]])
                yTrain = tf.constant([history[1]])

                print(xTrain)

                layer0 = tf.keras.layers.Dense(units=1, input_shape=(4,))

                model = tf.keras.Sequential([
                    layer0,
                ])

                model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))

                trainModel = model.fit(xTrain, yTrain, steps_per_epoch=1, epochs=5, verbose=False)

        # Prints information inside history (Information sent to tensorflow)
        if currentRun == 12 and timer <= 0:
            print(len(history))
            for num, h in enumerate(history, 0):
                if num % 5 == 0:
                    print("", end="")
                print(h[0], end=",\n")
            for h in history:
                print(h[1], end=",\n")

        for road in [road1, road2, road3, road4]:
            screen.blit(road[0], road[1])

        for car in ai:
            if not car.hit:
                car.takeAction()
                car.update()
                car.draw()
                car.drawLines(GREEN)
                car.drawContact(car.scan(currentRoad, preload), BLACK)
                car.raypoints = car.posToDis(car.scan(currentRoad, preload))
            else:
                car.draw(RED)
            gf.checkEvent(car)

        gen.blit()
        gf.drawCheckPointLines(screen)
        gf.drawImageBorders(screen)
        pygame.display.flip()
        clock.tick(60)


def test():
    xTrain = tf.constant(
        testData.trainingData()
    )
    yTrain = tf.constant(
        testData.trainingAnswers()
    )
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(80,))])
    model.compile(loss="mean_squared_logarithmic_error", optimizer=tf.keras.optimizers.SGD(0.1))

    trainModel = model.fit(xTrain, yTrain, steps_per_epoch=1, epochs=1000, verbose=False)

    prediction = model.predict([testData.testQuestions()])
    actual = testData.testAnswers()

    total = []
    print(len(testData.trainingData()))
    for num, p in enumerate(prediction, 0):
        total.append((p, actual[num]))
    total = sorted(total, key=lambda x: x[0])
    sectionSize = int(len(total)/3)
    low = total[0: sectionSize]
    med = total[sectionSize: sectionSize * 2]
    high = total[sectionSize * 2:]

    print("LOW NUMBERS!")
    for k in low:
        print("Expected: " + str(k[0]) + ", Actual: " + str(k[1]))

    print("MED NUMBERS!")
    for k in med:
        print("Expected: " + str(k[0]) + ", Actual: " + str(k[1]))

    print("HIGH NUMBERS!")
    for k in high:
        print("Expected: " + str(k[0]) + ", Actual: " + str(k[1]))

    model.evaluate([testData.testQuestions()], [testData.testAnswers()], verbose=1)
    # print(prediction)

# notmain()
test()
