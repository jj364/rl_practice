#!/usr/bin/env python3

import tkinter as tk
from tkinter import Canvas
from collections import deque
import random
import time

WIDTH = 200
HEIGHT = 200
DIFFICULTY = {"Easy": 400, "Medium": 200, "Hard": 100, "Ridiculous": 50}
BLOCK_SIZE = 10


class UI:
    def __init__(self):
        self.root = tk.Tk()

    def buttons(self, snake, canvas, score_str_var):
        self.root.after(snake.difficulty, snake.move, canvas, score_str_var, self.root)

        self.root.bind('<Left>', lambda event: snake.change_direction('l'))
        self.root.bind('<Right>', lambda event: snake.change_direction('r'))
        self.root.bind('<Up>', lambda event: snake.change_direction('u'))
        self.root.bind('<Down>', lambda event: snake.change_direction('d'))

        self.root.mainloop()


class Snake:
    def __init__(self, start_length=3, speed=10, body_colour='red', difficulty='Medium'):
        self.length = start_length
        self.speed = speed
        self.body = deque([])
        self.rect = deque([])
        self.body_colour = body_colour
        self.direction = 'r'
        self.food = None
        self.food_rect = None
        self.score = 0
        self.t0 = time.time()
        self.difficulty = DIFFICULTY[difficulty]

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

        if self.food_rect is not None:
            canvas.delete(self.food_rect)

        food = [random.randint(0,WIDTH/BLOCK_SIZE-1)*BLOCK_SIZE, random.randint(0,HEIGHT/BLOCK_SIZE-1)*BLOCK_SIZE]

        while food in self.body:
            food = [random.randint(0,WIDTH/BLOCK_SIZE)*BLOCK_SIZE, random.randint(0,HEIGHT/BLOCK_SIZE)*BLOCK_SIZE]

        x1 = food[0]; x2 = food[0] + BLOCK_SIZE
        y1 = food[1]; y2 = food[1] + BLOCK_SIZE

        self.food = food
        self.food_rect = canvas.create_rectangle(x1, y1, x2, y2, fill='green')

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

    def move(self, canvas, score_str_var, root, remove_end=True):
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

        if head_pos == self.food:
            self.gen_food(canvas)
            remove_end = False

            # update score
            self.score += 1
            score_str_var.set(str(self.score))
        else:
            remove_end = True

        # Call move again
        root.after(self.difficulty, self.move, canvas, score_str_var, root, remove_end)


def setup_game(difficulty):
    ui = UI()
    ui.root.rowconfigure(0, weight=1) # allow header expand vertically
    ui.root.columnconfigure(0, weight=1) # allow both header and footer expand horizontally

    header = tk.Frame(master=ui.root, height=30, bg="red")
    header.pack(fill=tk.X)

    canvas = Canvas(ui.root, height=HEIGHT, width=WIDTH, highlightthickness=3, highlightbackground="black")
    canvas.pack(fill=tk.X)

    footer = tk.Frame(master=ui.root, bg='#A5A5A5', height=30)
    footer.pack(fill=tk.X)
    score_str_var = tk.StringVar()
    live_score = tk.Label(ui.root, textvariable=score_str_var)
    live_score.place(relx=1.0, rely=1.0, anchor='se')
    score_str_var.set("0")

    snake = Snake(start_length=3, difficulty=difficulty)
    snake.create_snake(canvas)
    snake.gen_food(canvas)

    ui.buttons(snake, canvas, score_str_var)


if __name__ == "__main__":
    setup_game(difficulty="Ridiculous")