import pygame
import matplotlib.pyplot as plt   
from Pong_AI_Tools.paddle import Paddle
from Pong_AI_Tools.ball import Ball
from Standard.FNN import FeedForwardNeuralNetwork
from datasets.pong_data1 import train_x, train_y
import math

pygame.init()
pygame.font.init()
WIDTH, HEIGHT = 1280, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PONG AI LEARNING SIMLUATION")
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
    
# USE PONG DATA 1
def main():
    layers_dims = [2, 2]
    paddle_nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.075, 200, l2_regularization=False, binary_classification=False, multiclass_classification=True, regression=False, optimizer="gradient descent", learning_rate_decay=False, gradient_descent_variant="batch")
    paddle_nn.train()  # DONT TRAIN WHEN COLLECTING DATA, SET COLLECT-GAME-DATA TO TRUE AND COMMENT THIS

    paddle1 = Paddle(10, 150, WIDTH, HEIGHT, "L", 5, BLACK, "neural_network")
    paddle2 = Paddle(10, 150, WIDTH, HEIGHT, "R", 5, WHITE, paddle_nn)
    ball = Ball(30, 30, WIDTH, HEIGHT, GREEN)
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

        # WIN CONDITION
        current_streak, longest_streak = check_win(ball, current_streak, longest_streak)
        # MOVE BALL CONSTANTLY
        ball.update_position()

        if collect_game_data == True:
            paddle2.automate_movement(ball)
        if collect_game_data == False:
            paddle2.predict_movement( ball.rect.y/100, paddle1.rect.y/100 )



        if collect_game_data == True:
            game_inputs[0].append(ball.rect.y/100)
            game_inputs[1].append(paddle1.rect.y/100)
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


# INPUTS: ballX, ballY, distance
# OUTPUTS: up, down

# Function Complexity: For some tasks, especially simpler tasks or tasks with a linear relationship between inputs and outputs, a linear model (no hidden layers) might be sufficient to capture the underlying patterns in the data. Introducing a hidden layer can introduce unnecessary complexity, leading to overfitting or slower convergence.

# DEBUG STRATEGIES:
# ()- Collect more data for 2 inputs and train with same arcitecture and hyperparameters. 
# ()- More iterations.
# ()- Deeper network architecture
# ()- Failing to move down so randmoize the y-direction of ball at start