import pygame
import numpy as np
import os
import time
import math
import random

DISPLAY_WINDOW_X , DISPLAY_WINDOW_Y = (500,500)
DISPLAY_WINDOW = (DISPLAY_WINDOW_X,DISPLAY_WINDOW_Y)
CELL_SIZE_X , CELL_SIZE_Y = 10, 10
BLACK = (0,0,0)
FPS   = 15

class Snake:
	EMPTY_CELL    = 0
	BOUNDARY_CELL = 1
	SNAKE_CELL    = 2
	SNACK_CELL    = 3

	UP    = 0
	DOWN  = 1
	LEFT  = 2
	RIGHT = 3 

	def __init__(self,field_shape):
		if len(field_shape.shape) == (2):
			raise Exception('Failed to start game, check field_shape') 
		self.play_field  = np.zeros(field_shape)

		# Setting Boundary
		field_w , field_h = self.play_field.shape
		for i in range(field_h):
			self.play_field[0,i]         = self.BOUNDARY_CELL
		for i in range(field_h):
			self.play_field[field_w-1,i] = self.BOUNDARY_CELL
		for i in range(field_w):
			self.play_field[i,0]         = self.BOUNDARY_CELL
		for i in range(field_w):
			self.play_field[i,field_h-1] = self.BOUNDARY_CELL

		# Snake body
		self.snake_body_points = [(1,1)]
		# Snake dirction
		self.direction = self.RIGHT
		
		# Snack position
		self.get_new_snack()

	def get_new_snack(self):
		field_w , field_h = self.play_field.shape
		while True:
			self.snack_cell = (random.randint(2,field_w-2),
						  random.randint(2,field_h-2))
			if  self.snack_cell not in self.snake_body_points \
			and len(self.snake_body_points) < (field_w-1)*(field_h-1):
				break


	def draw(self,screen):
		#Draw region is reverse to np array struct
		cell_x , cell_y = screen.get_height()//self.play_field.shape[0] , screen.get_width()//self.play_field.shape[1]

		#Boundary
		WHITE = (255,255,255)
		RED   = (255,0,0)
		BLUE  = (0,0,255)

		pygame.draw.line(screen, WHITE , (0,0) , (0,screen.get_height()) , cell_y*2)
		pygame.draw.line(screen, WHITE , (0,screen.get_height()) , (screen.get_width(),screen.get_height()) , cell_x*2)
		pygame.draw.line(screen, WHITE , (0,0) , (screen.get_width(),0) , cell_x*2)
		pygame.draw.line(screen, WHITE , (screen.get_width(),0) , (screen.get_width(),screen.get_height()) , cell_y*2 )

		#Snake body points
		for x,y in self.snake_body_points: 
			pygame.draw.rect(screen,RED,(y*cell_y,x*cell_x,cell_y,cell_x))

		#Snake point
		pygame.draw.rect(screen,BLUE,(self.snack_cell[1]*cell_y,self.snack_cell[0]*cell_x,cell_y,cell_x))

	def is_update_ok(self,direction):

		if (
			self.direction in (self.UP,self.DOWN) 
			and  direction in (self.UP,self.DOWN)
		   ) or (
			self.direction in (self.RIGHT,self.LEFT)
			and  direction in (self.RIGHT,self.LEFT)
		   ):
				direction = self.direction

		last_point_x,last_point_y = self.snake_body_points[-1]
		if direction == self.UP:
			last_point_x -= 1
		elif direction == self.DOWN:
			last_point_x += 1
		elif direction == self.LEFT:
			last_point_y -= 1
		elif direction == self.RIGHT:
			last_point_y += 1	

		if self.play_field[last_point_x,last_point_y] == self.BOUNDARY_CELL or \
		   self.play_field[last_point_x,last_point_y] == self.SNAKE_CELL:
			return False , direction , (last_point_x,last_point_y) 
		else:
			return True , direction , (last_point_x,last_point_y)

	def update(self,direction):
		ok , self.direction, update_cell = self.is_update_ok(direction)

		# Render in Play Field
		if self.play_field[update_cell] == self.SNACK_CELL:  
			# New Snack position
			self.get_new_snack()
		else:
			cleanup_cell=self.snake_body_points.pop(0)
			self.play_field[cleanup_cell] = self.EMPTY_CELL

		self.snake_body_points.append(update_cell)
		self.play_field[update_cell]  = self.SNAKE_CELL

		# Render in Play Field
		self.play_field[self.snack_cell] = self.SNACK_CELL		

		#Game End
		if not ok:
			return False
		return True

def game_render_loop():

	#Game init
	pygame.init()

	#Screen Init
	pygame.display.set_caption("Snake")
	screen = pygame.display.set_mode(DISPLAY_WINDOW)

	# Game Init
	n_cells_y, n_cells_x = DISPLAY_WINDOW_X//CELL_SIZE_X, DISPLAY_WINDOW_Y//CELL_SIZE_Y
	snake = Snake(np.array([n_cells_x, n_cells_y]))

	# Loop Init
	running = True

	# Main Loop
	begin = time.time()
	clock = pygame.time.Clock()
	while running:
		# All keys pressed, is a dictonary of bools of all keys
		key=pygame.key.get_pressed()
		# All Updates
		if key[pygame.K_UP]:
			running = snake.update(snake.UP)
		elif key[pygame.K_DOWN]:
			running = snake.update(snake.DOWN)
		elif key[pygame.K_LEFT]:
			running = snake.update(snake.LEFT)
		elif key[pygame.K_RIGHT]:
			running = snake.update(snake.RIGHT)
		else:
			running = snake.update(snake.direction)
		# All Pygame Event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		#debugs
		#print(snake.snake_body_points)
		#os.system('clear')
		#print(snake.play_field)

		# All Draws
		'''
		1. Fill Screen With Black
		2. Fill Snake
		'''
		screen.fill(BLACK)
		snake.draw(screen)

		# For all the Draw operations 1 flip for frame
		pygame.display.flip()

		# Timing game loop
		clock.tick(FPS)
		now = time.time()
		#print("Loop : {0} seconds ".format(now - begin))
		begin = now

if __name__=="__main__":
	game_render_loop()
