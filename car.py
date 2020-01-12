import math
import pygame
import carFunctions as cf
from brain import Brain


class Car:
    def __init__(self, screen):
        self.screen = screen
        self.brain = Brain()

        self.x, self.y = 30.0, 130.0
        self.angle = -90  # need to add a limit so it doesnt overflow
        self.turn = 0

        self.points = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        self.raypoints = [25, 25, 25, 25, 25]
        self.mask = None
        self.hit = False
        self.checkpoint = 0
        self.timer = 0
        self.speed = 0
        self.maxSpeed = 1
        self.createMask()

        self.score = 0
        self.accelerate = True

    def getPos(self):
        return self.x, self.y

    def reset(self):
        self.x, self.y = 30.0, 130.0
        self.checkpoint = 0
        self.score = 0
        self.angle = -90
        self.hit = False
        self.timer = 0

    def endRun(self):
        pass

    def createMask(self):
        image = pygame.Surface((22, 22), pygame.SRCALPHA)
        points = [cf.calcPos((11, 11), cf.calcXY(self.angle - 10, 10)),
                  cf.calcPos((11, 11), cf.calcXY(self.angle + 10, 10)),
                  cf.calcPos((11, 11), cf.calcXY(self.angle + 155, 10)),
                  cf.calcPos((11, 11), cf.calcXY(self.angle + 205, 10))]
        pygame.draw.polygon(image, (100, 100, 200), points)
        self.mask = pygame.mask.from_surface(image)

    def printInfo(self):
        print("Position : (", self.x, ", ", self.y, ")", sep="")
        print("Angle : ", self.angle)
        print("Speed : ", self.speed)

    def takeAction(self):
        chosenAction = self.brain.pickAction(self.raypoints)
        if chosenAction <= 2:
            self.accelerate = False
            self.turn = chosenAction % 3
        else:
            self.accelerate = True
            self.turn = chosenAction % 3

    def update(self):
        if self.turn == 1:
            self.angle -= 5
        elif self.turn == 2:
            self.angle += 5

        if not self.hit:
            if self.accelerate and self.speed < self.maxSpeed:
                self.speed += .05
            elif self.speed > 0:
                self.speed -= .05
        else:
            self.speed = .05

        xMovement = round(math.cos(math.radians(self.angle)) * self.speed, 2)
        yMovement = round(math.sin(math.radians(self.angle)) * self.speed, 2)
        self.x += xMovement
        self.y += yMovement
        self.updateCar()

    def updateCar(self):
        self.points = [cf.calcPos((self.x, self.y), cf.calcXY(self.angle - 10, 10)),
                       cf.calcPos((self.x, self.y), cf.calcXY(self.angle + 10, 10)),
                       cf.calcPos((self.x, self.y), cf.calcXY(self.angle + 155, 10)),
                       cf.calcPos((self.x, self.y), cf.calcXY(self.angle + 205, 10))]
        self.createMask()

    def checkSpeed(self):
        if self.speed == 0:
            self.timer += 1
        if self.timer > 20:
            self.hit = True

    def draw(self, color=(100, 100, 200)):
        pygame.draw.polygon(self.screen, color, self.points)


    def drawLines(self, color):
        """Draws lines where the hit scan should be"""
        pygame.draw.line(self.screen, color, (self.x, self.y),
                         (cf.calcPos((self.x, self.y), cf.calcXY(self.angle - 60, 75))))
        pygame.draw.line(self.screen, color, (self.x, self.y),
                         (cf.calcPos((self.x, self.y), cf.calcXY(self.angle - 30, 75))))
        pygame.draw.line(self.screen, color, (self.x, self.y),
                         (cf.calcPos((self.x, self.y), cf.calcXY(self.angle + 0, 75))))
        pygame.draw.line(self.screen, color, (self.x, self.y),
                         (cf.calcPos((self.x, self.y), cf.calcXY(self.angle + 30, 75))))
        pygame.draw.line(self.screen, color, (self.x, self.y),
                         (cf.calcPos((self.x, self.y), cf.calcXY(self.angle + 60, 75))))

    def drawContact(self, points, color):
        for point in points:
            # print(point)
            pygame.draw.circle(self.screen, color, point, 4)

    def scan(self, road, preload=None):
        dot = pygame.Surface((1, 1), pygame.SRCALPHA)
        dot.fill((250, 0, 0))
        pixel = pygame.mask.from_surface(dot)
        points = []
        for i in range(5):
            for j in range(16):
                checkPos = cf.calcPos((self.x - road[1][0], self.y - road[1][1]), cf.calcXY(self.angle - 60 + (30*i), 5*j))
                contactPoint = road[2].overlap(pixel, (int(checkPos[0]), int(checkPos[1])))
                if contactPoint is not None:
                    lastScan = cf.calcPos((self.x, self.y), cf.calcXY(self.angle - 60 + (30 * i), 5 * j))
                    points.append((int(lastScan[0]), int(lastScan[1])))
                    break
                if j == 15:
                    lastScan = cf.calcPos((self.x, self.y), cf.calcXY(self.angle - 60 + (30 * i), 5 * j))
                    points.append((int(lastScan[0]), int(lastScan[1])))
        if preload is not None:
            for i in range(5):
                for j in range(16):
                    checkPos = cf.calcPos((self.x - preload[1][0], self.y - preload[1][1]), cf.calcXY(self.angle - 60 + (30*i), 5*j))
                    contactPoint = preload[2].overlap(pixel, (int(checkPos[0]), int(checkPos[1])))
                    if contactPoint is not None:
                        currentDis = math.sqrt((int(points[i][0] - self.x)**2) + (int(points[i][1] - self.y)**2))
                        lastScan = cf.calcPos((self.x, self.y), cf.calcXY(self.angle - 60 + (30 * i), 5 * j))
                        newDis = math.sqrt((int(lastScan[0] - self.x)**2) + (int(lastScan[1] - self.y)**2))
                        # print("Current: {}, New: {}" .format(currentDis, newDis))
                        if currentDis > newDis:
                            points[i] = ((int(lastScan[0]), int(lastScan[1])))
                            break
        return points

    def posToDis(self, scanPoints):
        distances = []
        for pos in scanPoints:
            currentDis = math.sqrt((int(pos[0] - self.x) ** 2) + (int(pos[1] - self.y) ** 2))
            distances.append(currentDis)
        return distances