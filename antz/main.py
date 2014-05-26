import random

import pygame
import pygame.draw
from pygame.locals import *

from antz import sim
from antz.chart import SolutionChartThread
from antz.graph_gen import GridGraphGenerator
from antz.gui_app import create_application
from antz.gui_sprites import AntSprite, WpSprite
from antz.gui_util import get_color, draw_solution_line, ColorDialog, default_font
from antz.sim import Waypoint, WaypointEdge

# general constants
LEFT = 1
RIGHT = 3
ANT_COUNT = 1000
MIN_BLUE = 70


def node_factory(name):
    if name == 'waypoint':
        return Waypoint
    else:
        raise ValueError('Node type %s does not exist' % name)


def edge_factory(name):
    if name == 'waypoint':
        return WaypointEdge
    else:
        raise ValueError('Edge type %s does not exist' % name)


node_config = {
    'nest': (get_color('green'), 255, 10, 10),
    'food': (get_color('red'), 255, 10, 10),
    'waypoint': (get_color('black'), 0, 2, 2)
}


def start_button_pressed(e, app, app_ctx):
    if not app_ctx.running:
        reset_button_pressed(e, app, app_ctx)
    app_ctx.paused = False


# noinspection PyUnusedLocal
def stop_button_pressed(e, app, app_ctx):
    app_ctx.paused = True


# noinspection PyUnusedLocal
def restart_button_pressed(e, app, app_ctx):
    if app_ctx.algorithm:
        app_ctx.running = False

        app_ctx.ant_sprites = pygame.sprite.Group()
        app_ctx.wp_sprites = pygame.sprite.Group()

        # create and set the runner
        app_ctx.runner = app_ctx.colony.create_runner(app_ctx.algorithm,
                                                      app_ctx.graph,
                                                      app_ctx.nest_node)

        # create the waypoint, nest, food sprites
        for node in app_ctx.graph.nodes:
            col, alpha, width, height = node_config.get(node.node_type)
            app_ctx.wp_sprites.add(
                WpSprite(app_ctx, node, (col, alpha), width, height))

        for ant in app_ctx.runner.ants:
            app_ctx.ant_sprites.add(AntSprite(app_ctx, ant, (89, 54, 99), 4, 4,
                                              color_dialog=app_ctx.ant_color_dialog))

        app_ctx.running = True


def reset_button_pressed(e, app, app_ctx):
    if app_ctx.algorithm:
        app_ctx.running = False

        graph_generator = GridGraphGenerator(app_ctx.anim_panel_x + app_ctx.anim_panel_width,
                                             app_ctx.anim_panel_y + app_ctx.anim_panel_height,
                                             10, node_factory, edge_factory, min_x=app_ctx.anim_panel_x,
                                             min_y=app_ctx.anim_panel_y, min_food_hops=100, max_food_hops=500)

        app_ctx.nest_node, app_ctx.food_node, app_ctx.graph = graph_generator()

        # precalculate lines between edges
        app_ctx.edge_lines = {}
        for edge in app_ctx.graph.edges:
            n1, n2 = edge.node_from, edge.node_to
            app_ctx.edge_lines[edge] = [(n1.x, n1.y), (n2.x, n2.y)]

        restart_button_pressed(e, app, app_ctx)

        app_ctx.running = True


# noinspection PyUnusedLocal
def on_show_grid_change(e, app, app_ctx, value):
    app_ctx.show_grid = value


# noinspection PyUnusedLocal
def on_show_grid_lines_change(e, app, app_ctx, value):
    app_ctx.show_grid_lines = value


class ApplicationContext(object):
    """
    Holds the whole state necessary for running the simulation
    """

    def __init__(self, solvers):
        self.running = False
        self.paused = True

        self.show_grid = False
        self.show_grid_lines = False

        self.graph = None
        self.colony = None
        self.solvers = solvers

        self.edge_lines = {}

        self.ant_sprites = None
        self.wp_sprites = None

        self.anim_panel_x = None
        self.anim_panel_y = None
        self.offset_screen_width = None
        self.offset_screen_height = None
        self.anim_panel_width = None
        self.anim_panel_height = None
        self.gui_panel_x = None
        self.gui_panel_y = None
        self.gui_panel_width = None
        self.gui_panel_height = None

        self.screen_width = None
        self.screen_height = None
        self.border_offset = None

        self.show_only_shortest = False
        self.done = False
        self.paint = False
        self.paint_erase = False
        self.state = 'stop'

        self.ant_color_dialog = None
        self.runner = None

        self.algorithm = None


def main():
    """
    Implements the mainloop
    """

    solvers = {
        sim.ShortestPathAlgorithm.TYPE: sim.ShortestPathAlgorithm()
    }

    ctx = ApplicationContext(solvers)

    # screen sizes
    ctx.screen_width = 1300
    ctx.screen_height = 700
    ctx.border_offset = 50

    ctx.anim_panel_x = ctx.border_offset
    ctx.anim_panel_y = ctx.border_offset
    ctx.offset_screen_width = ctx.screen_width - 2 * ctx.border_offset
    ctx.offset_screen_height = ctx.screen_height - 2 * ctx.border_offset
    ctx.anim_panel_width = ctx.offset_screen_width / 5 * 3
    ctx.anim_panel_height = ctx.offset_screen_height
    ctx.gui_panel_x = ctx.anim_panel_x + ctx.anim_panel_width + ctx.border_offset
    ctx.gui_panel_y = ctx.border_offset
    ctx.gui_panel_width = ctx.screen_width - 3 * ctx.border_offset - ctx.anim_panel_width
    ctx.gui_panel_height = ctx.offset_screen_height

    # state variables
    ctx.show_only_shortest = False
    ctx.done = False
    ctx.paint = False
    ctx.paint_erase = False
    ctx.state = 'stop'

    pygame.init()

    screen = pygame.display.set_mode([ctx.screen_width, ctx.screen_height], HWSURFACE | DOUBLEBUF | RESIZABLE)

    ctx.ant_color_dialog = ColorDialog('#000000')

    # CREATE THE ANT COLONY
    ctx.colony = sim.AntColony()

    application = create_application(ctx, ctx.screen_width, ctx.screen_height,
                                     ctx.gui_panel_x, ctx.gui_panel_y,
                                     ctx.gui_panel_width, ctx.gui_panel_height)

    application.listen_for('show_grid_lines', on_show_grid_lines_change)
    application.listen_for('show_grid', on_show_grid_change)
    application.listen_for('restart', restart_button_pressed)
    application.listen_for('reset', reset_button_pressed)
    application.listen_for('start', start_button_pressed)
    application.listen_for('stop', stop_button_pressed)

    solution_chart_thread = SolutionChartThread(ctx)
    solution_chart_thread.start()

    while ctx.done is False:
        for event in pygame.event.get():  # User did something
            application.app.event(event)

            if event.type == pygame.QUIT:  # If user clicked close
                ctx.done = True  # Flag that we are done so we exit this loop
            elif event.type is KEYDOWN and event.key == K_ESCAPE:
                ctx.done = True
            elif event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.dict['size'], HWSURFACE | DOUBLEBUF | RESIZABLE)
                pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
                if ctx.running:
                    ctx.paint = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                if ctx.running:
                    ctx.paint = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT:
                if ctx.running:
                    ctx.paint_erase = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == RIGHT:
                if ctx.running:
                    ctx.paint_erase = False
            elif event.type == pygame.MOUSEMOTION:
                if ctx.running:
                    if ctx.paint:
                        # replace nodes with obstacle nodes
                        for s in ctx.wp_sprites.sprites():
                            if s.rect.collidepoint(event.pos):
                                s.set_obstacle(True)
                    if ctx.paint_erase:
                        # replace nodes with obstacle nodes
                        for s in ctx.wp_sprites.sprites():
                            if s.rect.collidepoint(event.pos):
                                s.set_obstacle(False)

        screen.fill(get_color('white'))

        if ctx.runner and ctx.running:
            if not ctx.paused:
                ctx.runner.move()

            # Clear the screen
            #
            if not ctx.show_only_shortest:
                for edge in ctx.runner.updated_edges:
                    lines = ctx.edge_lines[edge]
                    plevel = edge.pheromone_level(
                        ctx.colony.pheromone_kind('default'))
                    if plevel:
                        level = 255 - (plevel * 10000)
                        if level < MIN_BLUE:
                            level = MIN_BLUE
                        if level > 255:
                            level = 255

                        pygame.draw.lines(screen, (0, 0, level), False,
                                          lines, 10)

            if ctx.show_grid_lines:
                for edge in ctx.graph.edges:
                    lines = ctx.edge_lines[edge]
                    pygame.draw.lines(screen, (0, 0, 0), False, lines, 1)

            label = default_font().render('Round %d' % ctx.runner.rounds, 5, get_color('black'))
            screen.blit(label, (100, 15))

            # ant_sprites.color = dialog.rgb
            ctx.ant_sprites.update()
            ctx.wp_sprites.update()

            # Draw all the spites
            ctx.wp_sprites.draw(screen)

            if ctx.runner.best_solution:
                draw_solution_line(screen, ctx.runner.best_solution,
                                   color=(255, 69, 0), thickness=8)

                # draw_best_solution_text(screen, solver.best_solution,
                #                         len(solver.solutions))

            # for solution in solver.solutions[:10]:
            #     if solution != solver.best_solution:
            #         draw_solution_line(screen, solution, thickness=1, color=(255, 255, 250))

            if not ctx.show_only_shortest:
                ctx.ant_sprites.draw(screen)

        application.app.paint()

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
