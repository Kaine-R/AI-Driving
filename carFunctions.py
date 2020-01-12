import math


def calcX(angle):
    return math.cos(math.radians(angle))


def calcY(angle):
    return math.sin(math.radians(angle))


def calcXY(angle, mult=1):
    x = calcX(angle)
    y = calcY(angle)
    x, y = round(x, 2) * mult, round(y, 2) * mult
    return x, y


def calcPos(pos, offset):
    return pos[0] + offset[0], pos[1] + offset[1]

def wallCollision():
    pass