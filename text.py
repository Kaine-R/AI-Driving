import pygame


class text:
    def __init__(self, screen, msg, pos=(0, 0)):
        self.screen = screen
        self.font = pygame.font.SysFont(None, 24)
        self.rect = pygame.Rect(pos, (1, 1))
        self.image = None
        self.prep(msg)

    def prep(self, msg):
        self.image = self.font.render(str(msg), True, (250, 250, 250))

    def setPos(self, pos):
        self.rect.x, self.rect.y = pos[0], pos[1]

    def blit(self):
        self.screen.blit(self.image, self.rect)
