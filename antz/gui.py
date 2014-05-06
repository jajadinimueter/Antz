import time
import random

import pygame
import pygame.draw
from pygame.locals import *

from pgu import gui
from pgu.gui import button

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

stop_pressed = True

def start(self):
    global stop_pressed
    stop_pressed = False

def stop(self):
    global stop_pressed
    stop_pressed = True

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
        
    def open(self, *args, **kw):
        super(ColorDialog, self).open(*args, **kw)
        self._sel_val = sel.value
        sel.value = '---'

    def close(self, *args, **kw):
        super(ColorDialog, self).close(*args, **kw)
        sel.value = self._sel_val

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

        self.wp = wp
 
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.set_alpha(0)
        self.image.fill((255,255,255))  

        # Fetch the rectangle object that has the dimensions of the image
        # image.
        # Update the position of this object by setting the values 
        # of rect.x and rect.y
        self.rect = self.image.get_rect()

        self.rect.y = self.wp.y - self.rect.height / 2.0
        self.rect.x = self.wp.x - self.rect.width / 2.0

    def set_obstacle(self, obstacle):
        self.wp.obstacle = obstacle
        if obstacle:
            self.image.fill(green)
            self.image.set_alpha(255)
        else:
            self.image.set_alpha(0)
            self.image.fill((255,255,255))  


# INIT PYGAME
pygame.init()

screen_width=1300
screen_height=700
top_offset = 100
bottom_offset = 30

screen=pygame.display.set_mode([screen_width,screen_height])

ant_sprites = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
wp_sprites = pygame.sprite.Group()

done = False
clock = pygame.time.Clock()

def create_sprite(node):
    if node.TYPE == 'waypoint':
        return WpSprite(node, None, 5.0, 5.0)
    if node.TYPE == 'food':
        return FoodSprite(node, green, 5.0, 5.0)
    elif node.TYPE == 'nest':
        return NestSprite(node, gray, 5.0, 5.0)

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
        wp = sim.WaypointEdge(n1, n2)
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
                if isinstance(sprite, WpSprite):
                    wp_sprites.add(sprite)
                else:
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

def replace_node(i, j, nodes, cb):
    x = nodes[i][j]
    p = cb(x)
    nodes[i][j] = p
    return p

def replace_random_node(nodes, cb):
    i, j = random_grid_location(nodes)    
    return replace_node(i,j,nodes,cb)

OFFSET = 10

def replace_food_node(i, j, nodes, cb):
    l = len(nodes)
    if i + OFFSET < l:
        i = i + OFFSET
    elif i - OFFSET > 0:
        i = i - OFFSET

    l  = len(nodes[i])
    if j + OFFSET < l:
        j += OFFSET
    elif j - OFFSET > 0:
        j -= OFFSET

    return replace_node(i,j,nodes,cb)

# setup the graph
grid_nodes = create_grid_nodes(
    screen_width, screen_height-top_offset-bottom_offset, 8, y=top_offset)
nest_i, nest_j = random_grid_location(grid_nodes)
nest = replace_node(nest_i, nest_j, grid_nodes, (lambda old: 
    sim.Nest(name='nest', x=old.x, y=old.y)))
food = replace_food_node(nest_i, nest_j, grid_nodes, (lambda old: 
    sim.Food(name='food', x=old.x, y=old.y)))
g = create_grid_graph(grid_nodes)

LEFT = 1
RIGHT = 3
ANT_COUNT = 100
# CREATE THE ANT COLONY
colony = sim.AntColony('colony-1')
pkind = colony.pheromone_kind('default')

shortest_path_behavior = sim.ShortestPathAlgorithm(g)

ants = sim.AntCollection(shortest_path_behavior)

def add_ant():
    ant = sim.Ant(colony, nest, shortest_path_behavior)
    sprite = AntSprite(ant, (89, 54, 99), 7, 7)
    ant_sprites.add(sprite)
    # all_sprites.add(sprite)
    ants.add(ant)

# CREATE THE ANTS
for i in range(0, ANT_COUNT):
    add_ant()

# precalculate lines between edges
edge_lines = {}
for edge in g.edges:
    n1, n2 = edge.node_from, edge.node_to
    edge_lines[edge] = [(n1.x, n1.y), (n2.x, n2.y)]

MIN_BLUE = 70
turn = 0

myfont = pygame.font.SysFont('monospace', 15)

# for ui controls
app = gui.App()

c = gui.Container(width=screen_width,height=screen_height)

e1 = gui.Button('Start')
e1.connect(gui.CLICK,start,None)
c.add(e1, screen_width-180, 13)

e2 = gui.Button('Stop')
e2.connect(gui.CLICK,stop,None)
c.add(e2, screen_width-100, 13)

e3 = gui.Button('Reset')
#e3.connect(gui.CLICK,,None)
c.add(e3, screen_width-180, 48)

e4 = gui.Button('Ant Color')
e4.connect(gui.CLICK,dialog.open,None)
c.add(e4, screen_width-100, 48)

show_only_shortest = False

def change_only_shortest_state(arg):
    global show_only_shortest
    btn, text = arg
    show_only_shortest = btn.value

def change_phero_dec(arg):
    inp, text = arg
    try:
        shortest_path_behavior.phero_dec = float(inp.value)
    except:
        pass

def change_num_ants(arg):
    inp, text = arg
    num = ANT_COUNT
    val = inp.value.strip()
    try:
        if not val:
            num = 0
        else:
            num = int(val)
    except:
        pass

    cur = len(ants)
    diff = cur - num
    if diff < 0:
        for i in range(0, -1 * diff):
            # we want more
            add_ant()
    elif diff > 0:
        removed = 0
        for ant_sprite in ant_sprites.sprites():
            if removed <= diff:
                ant_sprites.remove(ant_sprite)
                ants.remove(ant_sprite.ant)
                removed += 1
            else:
                break

def change_alpha(arg):
    try:
        val = float(arg.value.strip())
        shortest_path_behavior.alpha = val
    except:
        pass
        

def change_beta(arg):
    try:
        val = float(arg.value.strip())
        shortest_path_behavior.beta = val
    except:
        pass
        

lshorest_only = gui.Label('Shortest Only')
shortest_only = gui.Switch()
shortest_only.connect(gui.CHANGE, change_only_shortest_state, (shortest_only, 'Shortest Only'))
c.add(shortest_only, 400, 15)
c.add(lshorest_only, 270, 15)

alpha_label = gui.Label('Alpha')
alpha_field = gui.Input(value=shortest_path_behavior.alpha, size=10)
alpha_field.connect(gui.CHANGE, change_alpha, alpha_label)
c.add(alpha_field, 100, 50)
c.add(alpha_label, 0, 52)

beta_label = gui.Label('Beta')
beta_field = gui.Input(value=shortest_path_behavior.beta, size=10)
beta_field.connect(gui.CHANGE, change_beta, beta_label)
c.add(beta_field, 350, 50)
c.add(beta_label, 250, 52)

l_num_ants = gui.Label('Num Ants')
text_num_ants = gui.Input(value=ANT_COUNT, size=10)
text_num_ants.connect(gui.CHANGE, change_num_ants, (text_num_ants, 'Num Ants'))
c.add(text_num_ants, 600, 50)
c.add(l_num_ants, 500, 52)

l_phero_dec = gui.Label('Phero Decrease')
text_phero_dec = gui.Input(value=shortest_path_behavior.phero_dec, size=10)
text_phero_dec.connect(gui.CHANGE, change_phero_dec, (text_phero_dec, 'Phero Dec'))
c.add(text_phero_dec, 900, 50)
c.add(l_phero_dec, 750, 52)


sel = gui.Select(value='draw_obstacles')
sel.add('---', '---')
sel.add('Draw Obstacles', 'draw_obstacles')
c.add(sel, screen_width-400, 13)

app.init(c)

paint = False
paint_erase = False
app_events_dispatch = True

while done is False:
    for event in pygame.event.get(): # User did something
        app.event(event)
        
        if event.type == pygame.QUIT: # If user clicked close
            done = True # Flag that we are done so we exit this loop
        elif event.type is KEYDOWN and event.key == K_ESCAPE: 
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
            paint = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
            paint = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT:
            paint_erase = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == RIGHT:
            paint_erase = False
        elif event.type == pygame.MOUSEMOTION:
            if sel.value == 'draw_obstacles':
                if paint:
                    # replace nodes with obstacle nodes
                    for s in wp_sprites.sprites():
                        if s.rect.collidepoint(event.pos):
                            s.set_obstacle(True)
                if paint_erase:
                    # replace nodes with obstacle nodes
                    for s in wp_sprites.sprites():
                        if s.rect.collidepoint(event.pos):
                            s.set_obstacle(False)

    if not stop_pressed:
        ants.move()

    # Clear the screen
    screen.fill(white) 

    best_path = shortest_path_behavior.best_path
    best_length = shortest_path_behavior.best_path_length

    if not show_only_shortest:
        for edge in shortest_path_behavior.pheromone_edges:
            lines = edge_lines[edge]
            plevel = edge.pheromone_level(pkind)
            if plevel:
                # print(plevel)
                level = 255 - (plevel * 10000)**2
                if level < MIN_BLUE:
                    level = MIN_BLUE

                color = (0, 0, level)
                pygame.draw.lines(screen, color, False,
                    lines, 30)
    
    if best_path:
        # draw a line
        pygame.draw.lines(screen, (255,69,0), False, 
            [(n.x, n.y) for n in best_path], 6)

        # render text
        label = myfont.render('Best Length: %.2f' % best_length, 5, black)
        screen.blit(label, (400, screen_height - 20))

    label = myfont.render('Turn %d' % turn, 5, black)
    screen.blit(label, (100, 20))

    # ant_sprites.color = dialog.rgb 
    ant_sprites.update()

    # Draw all the spites
    wp_sprites.draw(screen)
    all_sprites.draw(screen)

    if not show_only_shortest:
        ant_sprites.draw(screen)
     
    # Limit to 20 frames per second
    clock.tick(60)

    app.paint()

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
    
    if not stop_pressed:
        turn += 1

pygame.quit()

