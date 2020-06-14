#!/usr/bin/env python3

import tkinter as tk
from collections import deque
import random

WIDTH = 200
HEIGHT = 200
TIME = 400
BLOCK_SIZE = 10


class Board:
    def __init__(self, master):
        self.whatevs = None


class Snake:
    def __init__(self, start_length=3, speed=10, body_colour='red'):
        self.length = start_length
        self.speed = speed
        self.body = deque([])
        self.rect = deque([])
        self.body_colour = body_colour
        self.direction = 'r'
        self.food = None

    def create_snake(self, canvas):
        head_pos = [int(WIDTH/2), int(HEIGHT)/2]

        for i in range(self.length):
            self.body.append([head_pos[0]-BLOCK_SIZE*i, head_pos[1]])

        for p in self.body:
            x1 = p[0];  x2 = p[0] + BLOCK_SIZE
            y1 = p[1];  y2 = p[1] + BLOCK_SIZE
            rect = canvas.create_rectangle(x1,y1,x2,y2, fill=self.body_colour)
            self.rect.append(rect)

    def gen_food(self, canvas):
        food = [random.randint(0,WIDTH/BLOCK_SIZE)*BLOCK_SIZE, random.randint(0,HEIGHT/BLOCK_SIZE)*BLOCK_SIZE]

        while food in self.body:
            food = [random.randint(0,WIDTH/BLOCK_SIZE)*BLOCK_SIZE, random.randint(0,HEIGHT/BLOCK_SIZE)*BLOCK_SIZE]

        x1 = food[0]; x2 = food[0] + BLOCK_SIZE
        y1 = food[1]; y2 = food[1] + BLOCK_SIZE
        self.food = canvas.create_rectangle(x1, y1, x2, y2, fill='green')


    def change_direction(self, direction):
        # Only allow 90 degree turns
        if direction == 'u' and self.direction == 'd':
            pass
        elif direction == 'd' and self.direction == 'u':
            pass
        elif direction == 'l' and self.direction == 'r':
            pass
        elif direction == 'r' and self.direction == 'l':
            pass
        else:
            self.direction = direction

    def velocity(self):
        if self.direction == 'u':
            v = [0, -BLOCK_SIZE]
        elif self.direction == 'd':
            v = [0, BLOCK_SIZE]
        elif self.direction == 'r':
            v = [BLOCK_SIZE, 0]
        elif self.direction == 'l':
            v = [-BLOCK_SIZE, 0]
        return v

    def check_head(self, h):
        # if head collides with itself or wall then quit!
        if h in self.body:
            return False
        elif h[0] not in range(0,WIDTH) or h[1] not in range(0,HEIGHT):
            return False
        else:
            return True

    def move(self, canvas, remove_end=True):
        # Move last element to front
        if remove_end:
            toe_to_head = self.rect.pop()
            self.body.pop()
            canvas.delete(toe_to_head)
        v = self.velocity()
        head_pos = [self.body[0][0]+v[0], self.body[0][1]+v[1]]
        if not self.check_head(head_pos):
            print("COLLISION...GAME OVER")
            quit()
        x1 = head_pos[0]; x2 = head_pos[0] + BLOCK_SIZE
        y1 = head_pos[1]; y2 = head_pos[1] + BLOCK_SIZE
        rect = canvas.create_rectangle(x1, y1, x2, y2, fill=self.body_colour)
        self.body.appendleft(head_pos)
        self.rect.appendleft(rect)

        root.after(TIME, self.move, canvas)


root = tk.Tk()
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

snake = Snake(start_length=5)
snake.create_snake(canvas)
snake.gen_food(canvas)

root.after(TIME, snake.move, canvas)

root.bind('<Left>', lambda event:snake.change_direction('l'))
root.bind('<Right>', lambda event:snake.change_direction('r'))
root.bind('<Up>', lambda event:snake.change_direction('u'))
root.bind('<Down>', lambda event:snake.change_direction('d'))

root.mainloop()