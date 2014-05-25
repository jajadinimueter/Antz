"""
GUI utilities go here
"""
from antz import sim

import pygame.font

from pgu import gui


COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'light_gray': (220, 220, 220)
}


def default_font():
    return pygame.font.SysFont('monospace', 15)


def get_color(name):
    """
    Return the color for the name
    """
    return COLORS.get(name, (0, 0, 0))


def calc_path_lines(path, line_cache=None):
    line_cache = line_cache or {}
    lines = line_cache.get(path, [])

    if not lines:
        l = len(path)
        for i, n in enumerate(path):
            lines.append((n.node_from.x, n.node_from.y))
            if i == l - 1:
                lines.append((n.node_to.x, n.node_to.y))
        line_cache[path] = lines

    return lines


# noinspection PyShadowingNames
def draw_solution_line(screen, solution, color=(210, 210, 210),
                       thickness=2, line_cache=None):
    if solution:
        solution, length = solution
        lines = calc_path_lines(solution, line_cache=line_cache)

        if lines:
            # draw a line
            pygame.draw.lines(screen, color, False, lines, thickness)


# noinspection PyShadowingNames
def draw_best_solution_text(screen, solution, num_solutions, position, font=None):
    solution, length = solution
    # render text
    label = font.render('Best Length: %.2f, Number of different solutions: %.2f'
                        % (length, num_solutions), 5, get_color('black'))
    screen.blit(label, position)


class ColorDialog(gui.Dialog):
    def __init__(self, value, on_open=None, on_close=None, **params):
        self._on_open = []
        self._on_close = []

        if on_open:
            self._on_open.append(on_open)

        if on_close:
            self._on_close.append(on_close)

        self.value = list(gui.parse_color(value))

        title = gui.Label('Color Picker')
        main = gui.Table()

        main.tr()

        self.color = gui.Color(self.value, width=64, height=64)
        main.td(self.color, rowspan=3, colspan=1)

        main.td(gui.Label(' Red: '), 1, 0)
        e = gui.HSlider(value=self.value[0], min=0, max=255, size=32, width=128, height=16)
        e.connect(gui.CHANGE, self.adjust, (0, e))
        main.td(e, 2, 0)

        main.td(gui.Label(' Green: '), 1, 1)
        e = gui.HSlider(value=self.value[1], min=0, max=255, size=32, width=128, height=16)
        e.connect(gui.CHANGE, self.adjust, (1, e))
        main.td(e, 2, 1)

        main.td(gui.Label(' Blue: '), 1, 2)
        e = gui.HSlider(value=self.value[2], min=0, max=255, size=32, width=128, height=16)
        e.connect(gui.CHANGE, self.adjust, (2, e))
        main.td(e, 2, 2)

        gui.Dialog.__init__(self, title, main, **params)

    def open(self, *args, **kw):
        super(ColorDialog, self).open(*args, **kw)
        for cb in self._on_open:
            cb(self)

    def close(self, *args, **kw):
        super(ColorDialog, self).close(*args, **kw)
        for cb in self._on_close:
            cb(self)

    def adjust(self, value):
        (num, slider) = value
        self.value[num] = slider.value
        self.color.repaint()
        self.send(gui.CHANGE)

    @property
    def rgb(self):
        return self.value

