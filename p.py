import pygame
import matplotlib.pyplot as plt   
from Pong_AI_Tools.paddle import Paddle
from Pong_AI_Tools.ball import Ball
from ScratchNetworks.NN import FeedForwardNeuralNetwork
from datasets.pong_data6 import train_x, train_y
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
    return math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))/100
    
# ballY paddleY ballXvel ballYvel distance
def main():
    layers_dims = [3, 5, 2] # try different DIMENSIONS, AND ADD MORE DATA
    paddle_nn = FeedForwardNeuralNetwork(train_x, train_y, layers_dims, 0.0075, 2500, binary_classification=False, multiclass_classification=True, regression=False, optimizer="gradient descent", learning_rate_decay=False, gradient_descent_variant="batch")
    paddle_nn.train()  # dont train when collecting data

    paddle1 = Paddle(10, 150, WIDTH, HEIGHT, "L", 5, BLACK, "neural_network")
    paddle2 = Paddle(10, 150, WIDTH, HEIGHT, "R", 5, WHITE, paddle_nn)
    ball = Ball(30, 30, WIDTH, HEIGHT, GREEN)
    collect_game_data = False    # TRUE to collect data
    game_inputs =  [[], [], []]
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
        if collect_game_data == False:  # ballY paddleY distance
            paddle2.predict_movement_6( ball.rect.y/100, paddle1.rect.y/100, calculate_distance(ball, paddle2) )

        # WIN CONDITION
        current_streak, longest_streak = check_win(ball, current_streak, longest_streak)
        # MOVE BALL CONSTANTLY
        ball.update_position()

        if collect_game_data == True: # ballY paddleY distance
            game_inputs[0].append(ball.rect.y/100)
            game_inputs[1].append(paddle1.rect.y/100)
            game_inputs[2].append(calculate_distance(ball, paddle2))

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
# (Success)- Collect 20 consective streak data yields 4 longest streak and paddle is moving up and down more.
# ()- Collect 30 consectutive streak data adn trainn using same network. 
# ()- Try differnet optimization methods.
# (?)- When adding only 1 hidden node paddle doesnt move.
# (F)- More iterations. causes math domain error
# (F)- Deeper network architecture. Causes math domain error. 

# PROBLEMS:
# ()- When adding only 1 hidden node paddle doesnt move.

# ballY paddleY distance

# INPUT DATA:
# ballY paddleY ballXvel ballYvel distance: RESULT
#   -       -       -                     : No movement
#   -       -       -       -       -     : Little movement
#