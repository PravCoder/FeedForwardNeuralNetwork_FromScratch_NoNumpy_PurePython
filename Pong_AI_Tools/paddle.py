import pygame
import numpy as np


class Paddle:

    def __init__(self, width, height, screen_width, screen_height, paddle_num, vel, color, neural_network):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_num = paddle_num
        if self.paddle_num == "L":
            self.rect = pygame.Rect(20, screen_height/2, width, height)
        if self.paddle_num == "R":
            self.rect = pygame.Rect(screen_height - height +600, screen_height/2, width, height)
        self.color = color
        self.vel = vel
        self.neural_network = neural_network
        self.fitness_score = 0
        self.status = "down"

    # Human controls
    def move(self, keys_pressed):
        if self.paddle_num == "L":
            if keys_pressed[pygame.K_w] and self.rect.y - self.vel > 0:
                self.rect.y -= self.vel
            if keys_pressed[pygame.K_s] and self.rect.y + self.vel + self.rect.height < self.screen_height:
                self.rect.y += self.vel
        if self.paddle_num == "R":
            if keys_pressed[pygame.K_UP] and self.rect.y - self.vel > 0:
                self.rect.y -= self.vel
            if keys_pressed[pygame.K_DOWN] and self.rect.y + self.vel + self.rect.height < self.screen_height:
                self.rect.y += self.vel

    # Hard coded control
    def automate_movement(self, ball):
        while self.rect.y < ball.rect.y:
            self.rect.y += self.vel
            self.status = "up"
        while self.rect.y > ball.rect.y:
            self.rect.y -= self.vel
            self.status = "down"
        
    # Neural network control
    def predict_movement(self, bally, paddley):

        # FNN.PY Network Classification
        inputs = [[bally], [paddley]]
        pred_i, predictions = self.neural_network.predict(inputs, [[0],[0]])
        #print("preds: " +str(predictions))
        if pred_i == 1 and self.rect.y - self.vel > 0:
            self.rect.y -= self.vel
        if pred_i == 0 and self.rect.y + self.vel + self.rect.height < self.screen_height:
            self.rect.y += self.vel


    def predict_movement_6(self, ballY, paddleY, distance): 
        # FNN.PY Network Classification
        inputs = [[ballY], [paddleY], [distance]]
        pred_i, predictions = self.neural_network.predict(inputs, [[0],[0]])

        if pred_i == 1 and self.rect.y - self.vel > 0:
            self.rect.y -= self.vel
        if pred_i == 0 and self.rect.y + self.vel + self.rect.height < self.screen_height:
            self.rect.y += self.vel

    def predict_movement_7(self, ballY, selfPaddleY):
        inputs = [[ballY], [selfPaddleY]]
        pred_i, predictions = self.neural_network.predict(inputs, [[0],[0]])
        if pred_i == 1 and self.rect.y - self.vel > 0:
            self.rect.y -= self.vel
        if pred_i == 0 and self.rect.y + self.vel + self.rect.height < self.screen_height:
            self.rect.y += self.vel

    def predict_movement_micrograd(self, ballX, ballY):
        inputs = [[ballX, ballY]]
        y_preds = self.neural_network.predic(inputs)
        print("Predictions: "+str(y_preds))
        

    def draw(self, win):
        pygame.draw.rect(win, self.color, self.rect)



# DEBUG STREGIES:
# ()- Implement regression to predict the y-coordinate of the paddle given the bally,paddley