import pygame
import numpy as np
import os
import time
import math
import random

DISPLAY_WINDOW_X , DISPLAY_WINDOW_Y = (100,100)
DISPLAY_WINDOW = (DISPLAY_WINDOW_X,DISPLAY_WINDOW_Y)
CELL_SIZE_X , CELL_SIZE_Y = 10, 10
BLACK = (0,0,0)
FPS   = 5

n_cells_y, n_cells_x = DISPLAY_WINDOW_X//CELL_SIZE_X, DISPLAY_WINDOW_Y//CELL_SIZE_Y

class Snake:
	EMPTY_CELL        = 0			# Empty cells needs to be zero because of init
	BOUNDARY_CELL      = -1

	SNAKE_TAIL_CELL    = -1
	SNAKE_BODY_CELL    = -1
	SNAKE_HEAD_CELL    = 1
	SNACK_CELL         = 2

	SNAKE_CELLS        = (SNAKE_TAIL_CELL,SNAKE_BODY_CELL,SNAKE_HEAD_CELL)

	UP    = 0
	DOWN  = 1
	LEFT  = 2
	RIGHT = 3
	DIRECTION_ARRAY = [UP,DOWN,LEFT,RIGHT]

	LIMITTED_STEPS = 50

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
		self.snake_body_points = [(1,1),(1,2)]
		self.play_field[self.snake_body_points[-1]] = self.SNAKE_HEAD_CELL
		self.play_field[self.snake_body_points[0]]  = self.SNAKE_TAIL_CELL 
		# Snake dirction
		self.direction = self.RIGHT
		
		# Snack position
		self.snack_cell = (None,None)
		self.get_new_snack()

		# Game info
		self.game_running = True

		#reward
		self.reward = 0

		#Step count init
		self.limited_steps = 0 

		#Is this game played by player
		self.is_player_game = False

	def reverse_dir(self):
		if self.direction == self.UP:
			return self.DOWN
		if self.direction == self.DOWN:
			return self.UP
		if self.direction == self.LEFT:
			return self.RIGHT
		if self.direction == self.RIGHT:
			return self.LEFT


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

		# Colours
		WHITE = (255,255,255)
		RED   = (255,0,0)
		BLUE  = (0,0,255)
		GREEN = (0,255,0)
		MAGENTA = (255,0,255)

		#Boundary
		pygame.draw.line(screen, WHITE , (0,0) , (0,screen.get_height()) , cell_y*2)
		pygame.draw.line(screen, WHITE , (0,screen.get_height()) , (screen.get_width(),screen.get_height()) , cell_x*2)
		pygame.draw.line(screen, WHITE , (0,0) , (screen.get_width(),0) , cell_x*2)
		pygame.draw.line(screen, WHITE , (screen.get_width(),0) , (screen.get_width(),screen.get_height()) , cell_y*2 )

		#Snake body points
		for x,y in self.snake_body_points[1:-1]: 
			pygame.draw.rect(screen,RED,(y*cell_y,x*cell_x,cell_y,cell_x))

		x,y = self.snake_body_points[0]
		pygame.draw.rect(screen,MAGENTA,(y*cell_y,x*cell_x,cell_y,cell_x))

		x,y = self.snake_body_points[-1]
		pygame.draw.rect(screen,GREEN,(y*cell_y,x*cell_x,cell_y,cell_x))

		#Snake point
		pygame.draw.rect(screen,BLUE,(self.snack_cell[1]*cell_y,self.snack_cell[0]*cell_x,cell_y,cell_x))

	def is_update_ok(self,direction):
		if self.is_player_game:
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
		   self.play_field[last_point_x,last_point_y] in self.SNAKE_CELLS:
			return False , direction , (last_point_x,last_point_y) 
		else:
			return True , direction , (last_point_x,last_point_y)

	def update_with_one_hot(self,one_hot):
		if len(one_hot) != len(self.DIRECTION_ARRAY):
			raise Exception('Error: onehot wrong for Snake',one_hot)
		return self.update(self.DIRECTION_ARRAY[np.argmax(one_hot)])

	def update(self,direction):
		# No updates after game is finished
		if not self.game_running:
			return False

		ok , self.direction, update_cell = self.is_update_ok(direction)

		# Render in Play Field, this order is very strict
		head_cell=self.snake_body_points[-1]
		self.play_field[head_cell]  = self.SNAKE_BODY_CELL
		self.snake_body_points.append(update_cell)

		if self.play_field[update_cell] == self.SNACK_CELL:  
			# New Snack position
			self.get_new_snack()
			self.reward = 50
		else:
			cleanup_cell=self.snake_body_points.pop(0)
			tail_cell   =self.snake_body_points[-1]
			self.play_field[cleanup_cell] = self.EMPTY_CELL
			self.play_field[tail_cell]   = self.SNAKE_TAIL_CELL

		self.play_field[update_cell]  = self.SNAKE_HEAD_CELL
		self.play_field[self.snack_cell] = self.SNACK_CELL	

		# Rewards
		self.reward -= 1

		# Step count
		self.limited_steps +=1 
		if self.limited_steps > self.LIMITTED_STEPS:
			ok = False 

		#Game End
		if not ok:
			self.reward -= 25
			self.game_running = False
			return False
		self.game_running = True
		return True

	def reset(self):
		self.__init__(np.array(self.play_field.shape))

		return self.get_state()

	def get_state(self):
		curr_state = self.play_field
		return np.array(curr_state.reshape([*curr_state.shape,1]))

	def get_reward(self):

		return  self.reward

def game_render_loop():

	#Game init
	pygame.init()

	#Screen Init
	pygame.display.set_caption("Snake")
	screen = pygame.display.set_mode(DISPLAY_WINDOW)

	# Game Init
	snake = Snake(np.array([n_cells_x, n_cells_y]))
	#snake.is_player_game = True

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
			pass
			#running = snake.update(snake.direction)
		# All Pygame Event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		#debugs
		#print(snake.snake_body_points)
		#os.system('clear')
		#print(snake.play_field)
		#print(snake.get_metrics())
		print(snake.get_reward())
		#if running == False:
		#	snake.reset()
		#	running = True

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

# This method requires tensorflow and loaded model
def game_predict_loop(model):
	#Game init
	pygame.init()

	#Screen Init
	pygame.display.set_caption("Snake")
	screen = pygame.display.set_mode(DISPLAY_WINDOW)

	# Game Init
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
		state = snake.get_state()
		action = model.predict(state.reshape([-1,*state.shape]))[0]
		running = snake.update_with_one_hot(action)
		
		#	running = snake.update(snake.direction)
		# All Pygame Event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		#debugs
		#print(snake.snake_body_points)
		#os.system('clear')
		#print(snake.play_field)
		#print(snake.get_metrics())
		#print(snake.get_reward())
		print(action)
		#if running == False:
		#	snake.reset()
		#	running = True

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
