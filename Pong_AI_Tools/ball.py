import pygame
import random


class Ball:

    def __init__(self, width, height, screen_width, screen_height, color):
        self.rect = pygame.Rect(screen_width/2, screen_height/2, width, height)
        self.color = color
        self.Xvel = 7
        self.Yvel = 5
        self.screen_width = screen_width
        self.screen_height = screen_height


    def update_position(self):
        self.rect.x += self.Xvel
        self.rect.y += self.Yvel

    def check_vertical_wall_collision(self):
        if self.rect.top <= 0 or self.rect.bottom >= self.screen_height:
            self.Yvel *= -1  

    def check_paddles_collision(self, paddle1, paddle2, count):
        if paddle1.rect.colliderect(self.rect):
            self.Yvel *= 1
            self.Xvel *= -1 
            paddle1.fitness_score += 1
            count += 1
        if paddle2.rect.colliderect(self.rect):
            self.Yvel *= 1
            self.Xvel *= -1 
            paddle2.fitness_score += 1       
            count += 1
        return count

    def draw(self, win):
        pygame.draw.rect(win, self.color, self.rect)