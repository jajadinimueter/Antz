import time
import random

import pygame
import pygame.draw

from antz import sim
from antz import graph
from antz.util import *

# Define some colors
black    = (   0,   0,   0)
white    = ( 255, 255, 255)
red      = ( 255,   0,   0)
blue     = ( 0,   0,   255)
green    = ( 0,   255,   0)
gray    = ( 220,   220,   220)

# This class represents the ball        
# It derives from the 'Sprite' class in Pygame
class AntSprite(pygame.sprite.Sprite):
     
    # Constructor. Pass in the color of the block, 
    # and its x and y position
    def __init__(self, ant, color, width, height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self) 
 
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
 
        # Fetch the rectangle object that has the dimensions of the image
        # image.
        # Update the position of this object by setting the values 
        # of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.ant = ant
        self.update()

    def update(self):
        node = self.ant.current_node
        if node:
            # Move the block down one pixel
            self.rect.y = node.y - self.rect.height / 2
            self.rect.x = node.x - self.rect.width / 2


class FoodSprite(pygame.sprite.Sprite):

    # Constructor. Pass in the color of the block, 
    # and its x and y position
    def __init__(self, food, color, width, height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self) 
 
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
 
        # Fetch the rectangle object that has the dimensions of the image
        # image.
        # Update the position of this object by setting the values 
        # of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.food = food
        self.update()

    def update(self):
        # Move the block down one pixel
        self.rect.y = self.food.y - self.rect.height / 2
        self.rect.x = self.food.x - self.rect.width / 2


class NestSprite(pygame.sprite.Sprite):

    # Constructor. Pass in the color of the block, 
    # and its x and y position
    def __init__(self, nest, color, width, height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self) 
 
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
 
        # Fetch the rectangle object that has the dimensions of the image
        # image.
        # Update the position of this object by setting the values 
        # of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.nest = nest
        self.update()

    def update(self):
        # Move the block down one pixel
        self.rect.y = self.nest.y - self.rect.height / 2
        self.rect.x = self.nest.x - self.rect.width / 2


class WpSprite(pygame.sprite.Sprite):

    # Constructor. Pass in the color of the block, 
    # and its x and y position
    def __init__(self, wp, color, width, height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self) 
 
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
 
        # Fetch the rectangle object that has the dimensions of the image
        # image.
        # Update the position of this object by setting the values 
        # of rect.x and rect.y
        self.rect = self.image.get_rect()
        self.wp = wp
        self.update()

    def update(self):
        # Move the block down one pixel
        self.rect.y = self.wp.y - self.rect.height / 2
        self.rect.x = self.wp.x - self.rect.width / 2


pygame.init()

screen_width=700
screen_height=500

screen=pygame.display.set_mode([screen_width,screen_height])

ant_sprites = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()

done = False
clock = pygame.time.Clock()

ANT_COUNT = 2000

# CREATE THE ANT COLONY
colony = sim.AntColony('colony-1')
pkind = colony.pheromone_kind('default')
shortest_path_behavior = sim.ShortestPathBehavior()
evaporate_strategy = sim.EvaporationStrategy(2)
ants = sim.AntCollection()

def create_sprite(node):
    if node.TYPE == 'waypoint':
        return WpSprite(node, gray, 5, 5)
    elif node.TYPE == 'food':
        return FoodSprite(node, blue, 10, 10)
    elif node.TYPE == 'nest':
        return NestSprite(node, green, 10, 10)

def create_grid_nodes(width, height, square_size):
    # creates a graph which is a grid
    ss = square_size
    x, y = 0, 0

    nodes = []

    cur_nodes = []

    while True:
        while True:
            node = sim.Waypoint(x=x, y=y)
            cur_nodes.append(node)
            if x > width:
                break
            x += square_size
        
        nodes.append(cur_nodes)
        cur_nodes = []

        if y > height:
            break

        x = 0
        y += square_size

    return nodes

def create_grid_graph(nodes):
    g = graph.Graph()
    def create_waypoint(n1, n2):
        wp = sim.WaypointEdge(n1, n2, 
            evaporation_strategy=evaporate_strategy)
        g.add_edge(wp)
        return wp
    yl = len(nodes)
    for i, xlist in enumerate(nodes):
        # make connections from left to right and
        # from top to bottom
        l = len(xlist)
        for j, a in enumerate(xlist):
            all_sprites.add(create_sprite(a))
            if j+1 < l:
                b = xlist[j+1]
                # create the wayfucker
                if a and b:
                    create_waypoint(a, b)
            if i+1 < yl:
                ylist = nodes[i+1]
                b = ylist[j]
                if a and b:
                    create_waypoint(a, b)
    return g

def random_grid_location(nodes):
    xlen = len(nodes[0])
    i = random.randrange(2, len(nodes)-2)
    j = random.randrange(2, xlen-2)
    return i, j

def replace_random_node(nodes, cb):
    i, j = random_grid_location(nodes)    
    x = nodes[i][j]
    p = cb(x)
    nodes[i][j] = p
    return p

# setup the graph
grid_nodes = create_grid_nodes(screen_width, screen_height, 15)
print('nodes created!')
nest = replace_random_node(grid_nodes, (lambda old: 
    sim.Nest(name='nest', x=old.x, y=old.y)))
food = replace_random_node(grid_nodes, (lambda old: 
    sim.Food(name='food', x=old.x, y=old.y)))
print('create graph')
g = create_grid_graph(grid_nodes)

# CREATE THE ANTS
for i in range(0, ANT_COUNT):
    ant = sim.Ant(colony, nest, shortest_path_behavior)
    sprite = AntSprite(ant, black, 7, 7)
    ant_sprites.add(sprite)
    all_sprites.add(sprite)
    ants.add(ant)

# precalculate lines between edges
edge_lines = []
for edge in g.edges:
    n1, n2 = edge.node_from, edge.node_to
    edge_lines.append((edge, [(n1.x, n1.y), (n2.x, n2.y)]))


while done == False:
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done = True # Flag that we are done so we exit this loop
 
    ants.move()
    for edge in g.edges:
        edge.evaporate_pheromone()

    # Clear the screen
    screen.fill(white) 

    best_path = shortest_path_behavior.best_path
    best_length = shortest_path_behavior.best_path_length

    for edge, lines in edge_lines:
        plevel = edge.pheromone_level(pkind)
        if plevel:
            level = 100 + plevel
            if level > 255:
                level = 255
            color = (100, 100, level)
            pygame.draw.lines(screen, color, False,
                lines, 10)
        else:
            pygame.draw.lines(screen, (240, 240, 240), False,
                lines, 1)

    if best_path:
        # draw a lsine
        pygame.draw.lines(screen, black, False, 
            [(n.x, n.y) for n in best_path], 6)
        
        myfont = pygame.font.SysFont('monospace', 15)

        # render text
        label = myfont.render('Best Length: %.2f' % best_length, 5, black)
        screen.blit(label, (400, 20))

    # Get the current mouse position. This returns the position
    # as a list of two numbers.
    pos = pygame.mouse.get_pos()
     
    # Draw all the spites
    all_sprites.draw(screen)

    ant_sprites.update()
     
    # Limit to 20 frames per second
    # clock.tick(20)

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
 
pygame.quit()