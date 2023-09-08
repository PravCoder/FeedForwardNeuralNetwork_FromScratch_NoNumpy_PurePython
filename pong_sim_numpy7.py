import pygame
import numpy as np
import matplotlib.pyplot as plt   
from Pong_AI_Tools.paddle import Paddle
from Pong_AI_Tools.ball import Ball
from NumpyNetworks.FNN import FeedForwardNeuralNetwork
from datasets.pong_data7 import train_x, train_y
import math

pygame.init()
pygame.font.init()
WIDTH, HEIGHT = 1280, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEURAL NETWORK LEARNS TO PLAY PONG SIMULATION")
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
FPS = 60
BLACK = (0, 0, 0)
BROWN = (244,164,96)
GREEN = (50,205,50)
STREAK_FONT = pygame.font.SysFont("comicsans", 40)
LOST_FONT = pygame.font.SysFont("comicsans", 100)


def draw_window(ball, paddle1, paddle2, current_streak, longest_streak):

    STREAK_SURFACE = STREAK_FONT.render(str("Current Streak: " +str(current_streak)), False, (0, 0, 0))
    LONGEST_SURFACE = STREAK_FONT.render(str("Longest Streak: " +str(longest_streak)), False, (0, 0, 0))

    WIN.fill(BROWN)
    ball.draw(WIN)
    paddle1.draw(WIN)
    paddle2.draw(WIN)
    WIN.blit(STREAK_SURFACE, (300,0))
    WIN.blit(LONGEST_SURFACE, (800,0))
    draw_line()
    pygame.display.update()

def check_win(ball, current_streak, longest_streak):
    winner_text = ""
    if ball.rect.left <= 0:                    # if ball's left hits the left screen#
        winner_text = "PLAYER 2 WINS"
        longest_streak = max(current_streak, longest_streak)
        current_streak = 0
    if ball.rect.right >= WIDTH:               # if ball's right hits right screen#
        winner_text = "PLAYER 1 WINS"
        longest_streak = max(current_streak, longest_streak)
        current_streak = 0
    if winner_text != "":
        ball.rect.x = HEIGHT/2
        ball.rect.y = WIDTH/2
        #print("Winner: " + winner_text)
    return current_streak, longest_streak

def draw_line():
    pygame.draw.line(WIN, BLACK, (WIDTH/2, 0), (WIDTH/2, HEIGHT))


def calculate_distance(ball, paddle):
    x1, y1 = paddle.rect.x, paddle.rect.y
    x2, y2 = ball.rect.x, ball.rect.y
    return math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
    
def main():
    layers_dims = [2, 10, 2]
    paddle_nn = FeedForwardNeuralNetwork(np.array(train_x), np.array(train_y), layers_dims, 0.075, 9500, multiclass_classification=True)
    paddle_nn.train()  # dont train when collecting data

    paddle1 = Paddle(10, 150, WIDTH, HEIGHT, "L", 5, BLACK, "neural_network")
    paddle2 = Paddle(10, 150, WIDTH, HEIGHT, "R", 5, WHITE, paddle_nn)
    ball = Ball(30, 30, WIDTH, HEIGHT, GREEN, is_random=True)
    collect_game_data = False    # TRUE to collect data
    game_inputs =  [[], []]
    game_outputs = [[], []]
    current_streak = 0
    longest_streak = 0

    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_SPACE]:
            print("train_x = " + str(game_inputs))
            print("train_y = " + str(game_outputs))
            run = False


        # BALL WALL COLLISION
        ball.check_vertical_wall_collision()    # reverses y-direction if it hits he top/bottom of screen#

        if collect_game_data == True:
            paddle2.automate_movement(ball)
        if collect_game_data == False:
            paddle2.predict_movement_7( ball.rect.y/100, paddle2.rect.y/100 ) # passing correct inputs into network

        # WIN CONDITION
        current_streak, longest_streak = check_win(ball, current_streak, longest_streak)
        # MOVE BALL CONSTANTLY
        ball.update_position()

        if collect_game_data == True:
            game_inputs[0].append(ball.rect.y/100)
            game_inputs[1].append(paddle2.rect.y/100)
        if collect_game_data == True:
            if paddle2.status == "up":
                game_outputs[0].append(1)
                game_outputs[1].append(0)
            if paddle2.status == "down":
                game_outputs[0].append(0)
                game_outputs[1].append(1)


        
        current_streak = ball.check_paddles_collision(paddle1, paddle2, current_streak)

        draw_window(ball, paddle1, paddle2, current_streak, longest_streak)
        
        #paddle1.move(keys_pressed)     # to move paddle 1 with human
        #paddle2.move(keys_pressed) 
        #paddle2.predict_movement(ball, paddle1)
        paddle1.automate_movement(ball)


    main()

if __name__ == "__main__":
    main()


""""
WELL PERFORMING HYPERPARAMETERS TABLE:
------------------------------------------------------------------
Artecture       Learing Rate        Iteartions      Dataset    Cost Status
[2, 10, 2]      0.075               9500            7           ocilation
[2, 10,5, 2]    0.1                 9500            7           ocilation



Status: performing better than human and is able to keep up with hard coded paddle

"""