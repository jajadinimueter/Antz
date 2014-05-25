import random

import pygame
import pygame.draw
from pygame.locals import *
from pgu import gui

from antz import sim
from antz.chart import SolutionChartThread
from antz.graph_gen import GridGraphGenerator
from antz.gui_app import create_application, ApplicationContext
from antz.gui_sprites import WpSprite, create_ant_sprite
from antz.gui_util import get_color, draw_solution_line, draw_best_solution_text, ColorDialog, default_font
from antz.sim import Waypoint, WaypointEdge

# general constants
LEFT = 1
RIGHT = 3
ANT_COUNT = 1000
MIN_BLUE = 70


def main():
    """
    Implements the mainloop
    """

    solvers = {
        sim.ShortestPathAlgorithm.TYPE: sim.ShortestPathAlgorithm
    }

    # screen sizes
    screen_width = 1300
    screen_height = 700
    border_offset = 50

    anim_panel_x = border_offset
    anim_panel_y = border_offset
    offset_screen_width = screen_width - 2 * border_offset
    offset_screen_height = screen_height - 2 * border_offset
    anim_panel_width = offset_screen_width / 5 * 3
    anim_panel_height = offset_screen_height
    gui_panel_x = anim_panel_x + anim_panel_width + border_offset
    gui_panel_y = border_offset
    gui_panel_width = screen_width - 3 * border_offset - anim_panel_width
    gui_panel_height = offset_screen_height

    # state variables
    show_only_shortest = False
    done = False

    pygame.init()

    screen = pygame.display.set_mode([screen_width, screen_height], HWSURFACE | DOUBLEBUF | RESIZABLE)

    ant_color_dialog = ColorDialog('#000000')
    ant_sprites = pygame.sprite.Group()
    wp_sprites = pygame.sprite.Group()

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

    graph_generator = GridGraphGenerator(anim_panel_x + anim_panel_width,
                                         anim_panel_y + anim_panel_height,
                                         10, node_factory, edge_factory, min_x=anim_panel_x,
                                         min_y=anim_panel_y, min_food_hops=100, max_food_hops=500)
    nest_node, food_node, graph = graph_generator()

    # CREATE THE ANT COLONY
    colony = sim.AntColony('colony-1')
    pkind = colony.pheromone_kind('default')

    solver = sim.ShortestPathAlgorithm(graph)

    ants = sim.AntCollection(solver)

    # CREATE THE ANTS
    for i in range(0, ANT_COUNT):
        ant, sprite = create_ant_sprite(colony, nest_node, solver,
                                        color_dialog=ant_color_dialog)
        ant_sprites.add(sprite)
        ants.add(ant)

    node_colors = {
        'nest': (get_color('green'), 255),
        'food': (get_color('red'), 255),
        'waypoint': (get_color('white'), 0)
    }

    # create the waypoint, nest, food sprites
    for node in graph.nodes:
        wp_sprites.add(WpSprite(node, node_colors.get(node.node_type), 10, 10))

    # precalculate lines between edges
    edge_lines = {}
    for edge in graph.edges:
        n1, n2 = edge.node_from, edge.node_to
        edge_lines[edge] = [(n1.x, n1.y), (n2.x, n2.y)]

    app_context = ApplicationContext(solvers, solver=solver)
    application = create_application(app_context, screen_width, screen_height,
                                     gui_panel_x, gui_panel_y,
                                     gui_panel_width, gui_panel_height)

    solution_chart_thread = SolutionChartThread(solver)
    solution_chart_thread.start()

    while done is False:
        for event in pygame.event.get():  # User did something
            application.app.event(event)

            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            elif event.type is KEYDOWN and event.key == K_ESCAPE:
                done = True
            elif event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.dict['size'], HWSURFACE | DOUBLEBUF | RESIZABLE)
                pygame.display.flip()
                # elif event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
                #     paint = True
                # elif event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                #     paint = False
                # elif event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT:
                #     paint_erase = True
                # elif event.type == pygame.MOUSEBUTTONUP and event.button == RIGHT:
                #     paint_erase = False
                # elif event.type == pygame.MOUSEMOTION:
                #     if sel.value == 'draw_obstacles':
                #         if paint:
                #             # replace nodes with obstacle nodes
                #             for s in wp_sprites.sprites():
                #                 if s.rect.collidepoint(event.pos):
                #                     s.set_obstacle(True)
                #         if paint_erase:
                #             # replace nodes with obstacle nodes
                #             for s in wp_sprites.sprites():
                #                 if s.rect.collidepoint(event.pos):
                #                     s.set_obstacle(False)

        ants.move()

        # Clear the screen
        screen.fill(get_color('white'))
        #
        if not show_only_shortest:
            for edge in solver.pheromone_edges:
                lines = edge_lines[edge]
                plevel = edge.pheromone_level(pkind)
                if plevel:
                    # print(plevel)
                    level = 255 - (plevel * 10000)
                    # print('Level: %s, Plevel: %s' % (level, plevel))
                    if level < MIN_BLUE:
                        level = MIN_BLUE
                    if level > 255:
                        level = 255

                    pygame.draw.lines(screen, (0, 0, level), False,
                                      lines, 10)

        label = default_font().render('Round %d' % solver.rounds, 5, get_color('black'))
        screen.blit(label, (100, 15))

        # ant_sprites.color = dialog.rgb
        ant_sprites.update()

        # Draw all the spites
        wp_sprites.draw(screen)
        #
        if solver.best_solution:
            draw_solution_line(screen, solver.best_solution,
                               color=(255, 69, 0), thickness=8)

            # draw_best_solution_text(screen, solver.best_solution,
            #                         len(solver.solutions))

        # for solution in solver.solutions[:10]:
        #     if solution != solver.best_solution:
        #         draw_solution_line(screen, solution, thickness=1, color=(255, 255, 250))

        if not show_only_shortest:
            ant_sprites.draw(screen)

        application.app.paint()

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
