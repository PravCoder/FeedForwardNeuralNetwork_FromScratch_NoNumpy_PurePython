import pygame
import sys

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Neural Network Visualization")

# Sample architecture (number of neurons in each layer)
architecture = [2, 3,10,10,10,10, 8, 3]

# Define node radius and gap between layers
NODE_RADIUS = 20
LAYER_GAP = SCREEN_WIDTH // (len(architecture) + 1)

def draw_neural_network(architecture):
    for i, layer_size in enumerate(architecture):
        x = (i + 1) * LAYER_GAP
        y_step = SCREEN_HEIGHT / (layer_size + 1)
        
        for j in range(layer_size):
            y = (j + 1) * y_step
            pygame.draw.circle(screen, BLACK, (x, int(y)), NODE_RADIUS)
            
            # Connect to the next layer
            if i < len(architecture) - 1:
                next_layer_size = architecture[i + 1]
                for k in range(next_layer_size):
                    next_x = (i + 2) * LAYER_GAP
                    next_y_step = SCREEN_HEIGHT / (next_layer_size + 1)
                    next_y = (k + 1) * next_y_step
                    pygame.draw.line(screen, BLACK, (x + NODE_RADIUS, int(y)), (next_x - NODE_RADIUS, int(next_y)), 2)

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(WHITE)
        draw_neural_network(architecture)
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
