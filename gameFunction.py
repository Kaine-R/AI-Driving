import math, sys
import pygame

def checkEvent(ai):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            keyDownEvents(event, ai)
        if event.type == pygame.KEYUP:
            keyUpEvents(event, ai)
        elif event.type == pygame.QUIT:
            sys.exit()

def checkSpace():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                return 100000
    return 0

def keyDownEvents(event, ai):
    if event.key == pygame.K_LEFT:
        ai.turn = 1
    elif event.key == pygame.K_RIGHT:
        ai.turn = 2
    elif event.key == pygame.K_DOWN:
        ai.maxSpeed -= .1
    elif event.key == pygame.K_UP:
        ai.maxSpeed += .1


def keyUpEvents(event, ai):
    if event.key == pygame.K_LEFT:
        ai.turn = 0
    elif event.key == pygame.K_RIGHT:
        ai.turn = 0

def setMap():
    road1Pic = pygame.image.load("road1.png").convert_alpha()
    road1Rect = road1Pic.get_rect()
    road1Mask = pygame.mask.from_surface(road1Pic)
    road2Pic = pygame.image.load("road2.png").convert_alpha()
    road2Rect = road2Pic.get_rect()
    road2Rect.x = 300
    road2Mask = pygame.mask.from_surface(road2Pic)
    road3Pic = pygame.image.load("road3.png").convert_alpha()
    road3Rect = road3Pic.get_rect()
    road3Rect.x, road3Rect.y = 300, 225
    road3Mask = pygame.mask.from_surface(road3Pic)
    road4Pic = pygame.image.load("road4.png").convert_alpha()
    road4Rect = road4Pic.get_rect()
    road4Rect.y = 225
    road4Mask = pygame.mask.from_surface(road4Pic)

    return (road1Pic, road1Rect, road1Mask), \
           (road2Pic, road2Rect, road2Mask), \
           (road3Pic, road3Rect, road3Mask), \
           (road4Pic, road4Rect, road4Mask)


def drawCheckPointLines(screen):
    pygame.draw.line(screen, (60, 120, 120), (66, 0), (66, 65))
    pygame.draw.line(screen, (60, 120, 120), (150, 150), (150, 220))
    pygame.draw.line(screen, (60, 120, 120), (240, 70), (315, 70))
    pygame.draw.line(screen, (60, 120, 120), (315, 150), (315, 220))
    pygame.draw.line(screen, (60, 120, 120), (400, 0), (400, 65))
    pygame.draw.line(screen, (60, 120, 120), (515, 170), (600, 170))
    pygame.draw.line(screen, (60, 120, 120), (505, 365), (580, 365))
    pygame.draw.line(screen, (60, 120, 120), (355, 365), (420, 365))
    pygame.draw.line(screen, (60, 120, 120), (310, 250), (310, 295))
    pygame.draw.line(screen, (60, 120, 120), (115, 365), (150, 365))
    pygame.draw.line(screen, (60, 120, 120), (270, 380), (345, 380))
    pygame.draw.line(screen, (60, 120, 120), (115, 415), (115, 450))
    pygame.draw.line(screen, (60, 120, 120), (0, 375), (60, 375))
    pygame.draw.line(screen, (60, 120, 120), (0, 150), (60, 150))

def checkpointCollision(car):
    checkpoint = [(66, 0, 66, 65), (150, 150, 150, 220),
                  (240, 70, 315, 70), (315, 150, 315, 220),
                  (400, 0, 400, 65), (515, 170, 600, 170),
                  (505, 365, 580, 365), (355, 365, 420, 365),
                  (310, 250, 310, 295), (115, 365, 150, 365),
                  (270, 380, 345, 380), (115, 415, 115, 450),
                  (0, 375, 60, 375), (0, 150, 60, 150)]

    if checkpoint[car.checkpoint][0]-2 < car.x < checkpoint[car.checkpoint][2]+2:
        if checkpoint[car.checkpoint][1]-2 < car.y < checkpoint[car.checkpoint][3]+2:
            car.checkpoint = (car.checkpoint + 1) % 15
            car.score += 100



def drawImageBorders(screen):
    pygame.draw.line(screen, (20, 60, 20), (300, 0), (300, 450))
    pygame.draw.line(screen, (20, 60, 20), (0, 225), (600, 225))

def selectRoad(ai, roads):
    if int(ai[0]) < 300:
        if int(ai[1]) < 225:
            return roads[0], 0
        return roads[3], 3
    else:
        if int(ai[1]) < 225:
            return roads[1], 1
        return roads[2], 2


def preloadRoad(ai, roads, roadNum):
    if 225 < ai[0] < 375:  # If car/ai is near next road(Hor.), sends back for preload
        if roadNum == 0:
            return roads[1]
        return roads[3]
    elif 150 < ai[1] < 300: # If car/ai is near next road(Ver.), sends back for preload
        if roadNum == 1:
            return roads[2]
        return roads[0]
    return None
