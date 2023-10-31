import pygame
import os
pygame.init()
pygame.font.init()
WIDTH, HEIGHT = 1280, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PONG AI REINFORCEMENT LEARNING SIMLUATION")
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
FPS = 60
VEL = 5
BALL_WIDTH, BALL_HEIGHT = 30, 30
P_WIDTH, P_HEIGHT = 20, 150

X_SPEED = 7
Y_SPEED = 7
rally_count = 0

RALLY_FONT = pygame.font.SysFont("comicsans", 40)
LOST_FONT = pygame.font.SysFont("comicsans", 100)

BG_IMAGE = pygame.image.load(os.path.join("pong.images", "brown_bg.png"))
BG = pygame.transform.scale(BG_IMAGE, (WIDTH, HEIGHT))

ball = pygame.Rect(WIDTH/2, HEIGHT/2, BALL_WIDTH, BALL_HEIGHT)
paddle1 = pygame.Rect(20, HEIGHT/2, P_WIDTH, P_HEIGHT)
paddle2 = pygame.Rect(WIDTH - P_WIDTH -20, HEIGHT/2, P_WIDTH, P_HEIGHT)


def draw_window():
    WIN.blit(BG, (0, 0))
    rally_text = RALLY_FONT.render("Streak: " + str(rally_count), 1, RED)
    WIN.blit(rally_text, (100, 15))
    pygame.draw.rect(WIN, YELLOW, ball)
    pygame.draw.rect(WIN, WHITE, paddle1)
    pygame.draw.rect(WIN, WHITE, paddle2)
    pygame.display.update()

def draw_winner(text):
    draw_text = LOST_FONT.render(text, 1, RED)
    WIN.blit(draw_text, (400, 400))
    pygame.display.update()
    pygame.time.delay(3000)


def paddle1_movement(keys_pressed, paddle1):
    if keys_pressed[pygame.K_w] and paddle1.y - VEL > 0:
        paddle1.y -= VEL
    if keys_pressed[pygame.K_s] and paddle1.y + VEL + paddle1.height < HEIGHT:
        paddle1.y += VEL
def paddle2_movement(keys_pressed, paddle2):
    if keys_pressed[pygame.K_UP] and paddle2.y - VEL > 0:
        paddle2.y -= VEL
    if keys_pressed[pygame.K_DOWN] and paddle2.y + VEL + paddle2.height < HEIGHT:
        paddle2.y += VEL


def main():

    global X_SPEED
    global Y_SPEED
    global rally_count

    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        keys_pressed = pygame.key.get_pressed()

        ball.x += X_SPEED              # ball is constantly moving cus were adding to the x-value#
        ball.y += Y_SPEED              # ball is constantly moving cus were add to the y-value#
                                       # if negative-value is mutiplied to the speed, then it subtracts the corrdinates
                                       # sending in opposite direction#
        if ball.top <= 0 or ball.bottom >= HEIGHT:
            Y_SPEED *= -1                            # reverses y-direction if it hits he top/bottom of screen#

        winner_text = ""
        if ball.left <= 0:                    # if ball's left hits the left screen#
            winner_text = "PLAYER 2 WINS"
        if ball.right >= WIDTH:               # if ball's right hits right screen#
            winner_text = "PLAYER 1 WINS"
        if winner_text != "":
            draw_winner(winner_text)
            break

        if ball.colliderect(paddle1):
            X_SPEED *= -1                # reverses the x direction#
            Y_SPEED *= 1                 # keeps the y-direction same#
            rally_count += 1
        if ball.colliderect(paddle2):
            X_SPEED *= -1
            Y_SPEED *= 1
            rally_count += 1


        draw_window()
        paddle1_movement(keys_pressed, paddle1)
        paddle2_movement(keys_pressed, paddle2)

    main()


if __name__ == "__main__":
    main()


