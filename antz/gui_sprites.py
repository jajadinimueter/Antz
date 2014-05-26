"""
Holds the various sprites
"""
import random
from antz import sim

import pygame.sprite
from antz.gui_util import get_color


class AntSprite(pygame.sprite.Sprite):
    def __init__(self, ant, color, width, height, color_dialog=None):
        pygame.sprite.Sprite.__init__(self)

        self._ant = ant
        self._color = color
        self._color_dialog = color_dialog
        self._x_offset = random.random() * 4
        self._y_offset = random.random() * 4

        self.image = pygame.Surface([width, height])
        self.image.fill(self.get_color())

        self.rect = self.image.get_rect()
        self.update()

    def get_color(self):
        if self._color_dialog:
            return self._color_dialog.rgb
        else:
            return self._color

    def update(self):
        node = self._ant.current_node

        if node:
            # Move the block down one pixel
            self.rect.y = node.y - self.rect.height / 2.0 - self._x_offset
            self.rect.x = node.x - self.rect.width / 2.0 - self._y_offset

        self.image.fill(self.get_color())


class WpSprite(pygame.sprite.Sprite):
    def __init__(self, node, color, width, height):
        pygame.sprite.Sprite.__init__(self)
        self._node = node
        self._color, self._alpha = color
        self.image = pygame.Surface([width, height])
        self.image.set_alpha(self._alpha)
        self.image.fill(self._color)
        self.rect = self.image.get_rect()
        self.rect.y = self._node.y - self.rect.height / 2.0
        self.rect.x = self._node.x - self.rect.width / 2.0

    def set_obstacle(self, obstacle):
        if not self._node.nest and not self._node.food:
            self._node.obstacle = obstacle
            if obstacle:
                self.image.fill(get_color('green'))
                self.image.set_alpha(255)
            else:
                self.image.set_alpha(0)
                self.image.fill((255, 255, 255))

