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

g = graph.Graph()

nest = sim.Nest(name='nest', x=200, y=100)
wp1 = sim.Waypoint(name='wp-1', x=10, y=50)
wp2 = sim.Waypoint(name='wp-2', x=6, y=120)
wp3 = sim.Waypoint(name='wp-3', x=300, y=300)
wp4 = sim.Waypoint(name='wp-4', x=34, y=45)
wp5 = sim.Waypoint(name='wp-5', x=78, y=44)
food = sim.Food(name='food', x=100, y=150)

all_sprites.add(WpSprite(wp1, black, 5, 5))
all_sprites.add(WpSprite(wp2, black, 5, 5))
all_sprites.add(WpSprite(wp3, black, 5, 5))
all_sprites.add(WpSprite(wp4, black, 5, 5))
all_sprites.add(WpSprite(wp5, black, 5, 5))

evaporate_strategy = sim.EvaporationStrategy(amount=10)

def create_waypoint(n1, n2):
    wp = sim.WaypointEdge(n1, n2, 
        evaporation_strategy=evaporate_strategy)
    g.add_edge(wp)
    return wp

# we need to create a waypoint factory
create_waypoint(nest, wp1)
create_waypoint(nest, wp2)
create_waypoint(wp1, wp3)
create_waypoint(wp2, wp3)
create_waypoint(wp3, wp4)
create_waypoint(wp3, wp5)
create_waypoint(wp3, food)
create_waypoint(wp5, food)

colony = sim.AntColony('colony-1')
shortest_path_behavior = sim.ShortestPathBehavior()

ants = sim.AntCollection()

for i in range(0, 100):
    ant = sim.Ant(colony, nest, shortest_path_behavior)
    sprite = AntSprite(ant, green, 3, 3)
    ants.add(ant)
    ant_sprites.add(sprite)
    all_sprites.add(sprite)

all_sprites.add(NestSprite(nest, blue, 10, 10))
all_sprites.add(FoodSprite(food, red, 10, 10))

pkind = colony.pheromone_kind('default')

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
        color = (200, 200, 200)
        plevel = edge.pheromone_level(pkind)
        if plevel:
            blueness = min([plevel, 255])
            color = (20, 20, blueness)
        pygame.draw.lines(screen, color, False,
            lines, 4)

    if best_path:
        # draw a lsine
        pygame.draw.lines(screen, black, False, 
            [(n.x, n.y) for n in best_path], 3)
        
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