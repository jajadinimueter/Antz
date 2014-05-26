"""
Holds the various sprites
"""
import random
from antz import sim

import pygame.sprite
from antz.gui_util import get_color


class AntSprite(pygame.sprite.Sprite):
    def __init__(self, app_ctx, ant, color, width, height, color_dialog=None):
        pygame.sprite.Sprite.__init__(self)

        self._app_ctx = app_ctx
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
    def __init__(self, app_ctx, node, color, width, height):
        pygame.sprite.Sprite.__init__(self)
        self._node = node
        self._app_ctx = app_ctx
        self._color, self._alpha = color

        self._resize = self._node.obstacle

        self._orig_color, self._orig_alpha = self._color, self._alpha
        self._width = self._orig_width = width
        self._height = self._orig_height = height

        self._draw_surface(self._width, self._height)

    def set_obstacle(self, obstacle):
        if not self._node.nest and not self._node.food:
            self._node.obstacle = obstacle
            self._resize = not self._resize

    def _draw_surface(self, width, height):
        self.image = pygame.Surface([width, height])
        self.image.set_alpha(self._alpha)
        self.image.fill(self._color)
        self.rect = self.image.get_rect()
        self.rect.y = self._node.y - self.rect.height / 2.0
        self.rect.x = self._node.x - self.rect.width / 2.0

    def update(self):
        if not self._node.obstacle:
            self._width = self._orig_width
            self._height = self._orig_height
            if self._app_ctx.show_grid:
                self._alpha = 255
            else:
                self._alpha = self._orig_alpha
                self._color = self._orig_color
        else:
            self._alpha = 255
            self._color = get_color('green')
            self._width = 8
            self._height = 8

        if self._resize:
            self._draw_surface(self._width, self._height)

        self.image.set_alpha(self._alpha)
        self.image.fill(self._color)