import time
import random

import pygame
import pygame.draw
from pygame.locals import *

from pgu import gui

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


class ColorDialog(gui.Dialog):
    def __init__(self,value,**params):
        self.value = list(gui.parse_color(value))
        
        title = gui.Label('Color Picker')
        
        main = gui.Table()
        
        main.tr()
        
        self.color = gui.Color(self.value,width=64,height=64)
        main.td(self.color,rowspan=3,colspan=1)
        
        main.td(gui.Label(' Red: '),1,0)
        e = gui.HSlider(value=self.value[0],min=0,max=255,size=32,width=128,height=16)
        e.connect(gui.CHANGE,self.adjust,(0,e))
        main.td(e,2,0)
        
        main.td(gui.Label(' Green: '),1,1)
        e = gui.HSlider(value=self.value[1],min=0,max=255,size=32,width=128,height=16)
        e.connect(gui.CHANGE,self.adjust,(1,e))
        main.td(e,2,1)

        main.td(gui.Label(' Blue: '),1,2)
        e = gui.HSlider(value=self.value[2],min=0,max=255,size=32,width=128,height=16)
        e.connect(gui.CHANGE,self.adjust,(2,e))
        main.td(e,2,2)
                        
        gui.Dialog.__init__(self,title,main)
        
    @property
    def rgb(self):
        return self.value

    def adjust(self,value):
        (num, slider) = value
        self.value[num] = slider.value
        self.color.repaint()
        self.send(gui.CHANGE)
    

dialog = ColorDialog('#000000')
        

# This class represents the ball        
# It derives from the 'Sprite' class in Pygame
class AntSprite(pygame.sprite.Sprite):
     
    # Constructor. Pass in the color of the block, 
    # and its x and y position
    def __init__(self, ant, color, width, height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self) 
 
        self.color = color #dialog.rgb
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.fill(dialog.rgb)

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
            self.rect.y = node.y - self.rect.height / 2.0
            self.rect.x = node.x - self.rect.width / 2.0
        self.image.fill(dialog.rgb)


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
        self.rect.y = self.food.y - self.rect.height / 2.0
        self.rect.x = self.food.x - self.rect.width / 2.0


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
        self.rect.y = self.nest.y - self.rect.height / 2.0
        self.rect.x = self.nest.x - self.rect.width / 2.0


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
        self.rect.y = self.wp.y - self.rect.height / 2.0
        self.rect.x = self.wp.x - self.rect.width / 2.0


pygame.init()

screen_width=1024
screen_height=600
top_offset = 60

screen=pygame.display.set_mode([screen_width,screen_height])

ant_sprites = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()

done = False
clock = pygame.time.Clock()

ANT_COUNT = 1000
# CREATE THE ANT COLONY
colony = sim.AntColony('colony-1')
pkind = colony.pheromone_kind('default')
shortest_path_behavior = sim.ShortestPathBehavior()

class MyStrategy(object):
    def amount(self, current_amount):
        return current_amount / 1.01

evaporate_strategy = MyStrategy()
ants = sim.AntCollection(shortest_path_behavior)

def create_sprite(node):
    # if node.TYPE == 'waypoint':
    #     return WpSprite(node, gray, 5.0, 5.0)
    if node.TYPE == 'food':
        return FoodSprite(node, green, 25.0, 25.0)
    elif node.TYPE == 'nest':
        return NestSprite(node, gray, 25.0, 25.0)

def create_grid_nodes(width, height, square_size, x=0, y=0):
    # creates a graph which is a grid
    ss = square_size

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
            sprite = create_sprite(a)
            ylist = None
            if i+1 < yl:
                ylist = nodes[i+1]
            if sprite:
                all_sprites.add(sprite)
            if j+1 < l:
                # create the wayfucker
                create_waypoint(a, xlist[j+1])
                if ylist:
                    # diagonal
                    create_waypoint(a, ylist[j+1])
                    if j-1 >= 0:
                        create_waypoint(a, ylist[j-1])
            if ylist:
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
grid_nodes = create_grid_nodes(
    screen_width, screen_height-top_offset, 15, y=top_offset)
nest = replace_random_node(grid_nodes, (lambda old: 
    sim.Nest(name='nest', x=old.x, y=old.y)))
food = replace_random_node(grid_nodes, (lambda old: 
    sim.Food(name='food', x=old.x, y=old.y)))
g = create_grid_graph(grid_nodes)

# CREATE THE ANTS
for i in range(0, ANT_COUNT):
    ant = sim.Ant(colony, nest, shortest_path_behavior)
    sprite = AntSprite(ant, (89, 54, 99), 7, 7)
    ant_sprites.add(sprite)
    all_sprites.add(sprite)
    ants.add(ant)

# precalculate lines between edges
edge_lines = []
for edge in g.edges:
    n1, n2 = edge.node_from, edge.node_to
    edge_lines.append((edge, [(n1.x, n1.y), (n2.x, n2.y)]))

MIN_BLUE = 70
turn = 0

myfont = pygame.font.SysFont('monospace', 15)

# for ui controls
app = gui.App()

c = gui.Container(width=screen_width,height=screen_height)

e = gui.Button('Ant Color')
e.connect(gui.CLICK,dialog.open,None)
c.add(e, screen_width-100, 13)

app.init(c)

while done == False:
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done = True # Flag that we are done so we exit this loop
        elif event.type is KEYDOWN and event.key == K_ESCAPE: 
            done = True
        else:
            app.event(event)
 
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
            level = 255 - (plevel * 6)**1.2
            if level < MIN_BLUE:
                level = MIN_BLUE

            color = (0, 0, level)
            pygame.draw.lines(screen, color, False,
                lines, 20)
        # else:
        #     pygame.draw.lines(screen, (240, 240, 240), False,
        #         lines, 1)

    if best_path:
        # draw a line
        pygame.draw.lines(screen, (255,69,0), False, 
            [(n.x, n.y) for n in best_path], 6)
        

        # render text
        label = myfont.render('Best Length: %.2f' % best_length, 5, black)
        screen.blit(label, (400, 20))

    label = myfont.render('Turn %d' % turn, 5, black)
    screen.blit(label, (100, 20))

    # Get the current mouse position. This returns the position
    # as a list of two numbers.
    pos = pygame.mouse.get_pos()
     
    # ant_sprites.color = dialog.rgb 
    ant_sprites.update()

    # Draw all the spites
    all_sprites.draw(screen)
     
    # Limit to 20 frames per second
    # clock.tick(60)

    app.paint()

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
 
    turn += 1

pygame.quit()

