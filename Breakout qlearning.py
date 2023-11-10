import numpy as np
import pygame
from pygame.locals import *
import random

# Dimensiones de la ventana del juego
WIDTH, HEIGHT = 640, 480

# Definición de los colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Parámetros del algoritmo Q-learning
LEARNING_RATE = 0.7
DISCOUNT_FACTOR = 0.9

# Parámetros del juego
N_ROWS = 5
N_COLS = 10
PADDLE_WIDTH = 60
PADDLE_HEIGHT = 10
BALL_RADIUS = 10
BALL_SPEED = 30

class QLearningAgent:
    def __init__(self, n_actions, learning_rate, discount_factor):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((N_ROWS * N_COLS, n_actions))

    def get_action(self, state):
        if random.uniform(0, 1) < 0.1:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        max_q_value = np.max(self.q_table[next_state])
        current_q_value = self.q_table[state][action]
        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * max_q_value)
        self.q_table[state][action] = new_q_value

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Atari Breakout")

    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 36)
    iteration = 0

    agent = QLearningAgent(3, LEARNING_RATE, DISCOUNT_FACTOR)

    max_rewards = 0  # Recompensa máxima alcanzada

    for _ in range(500):

        paddle_x = WIDTH // 2 - PADDLE_WIDTH // 2
        ball_x = random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS)
        ball_y = HEIGHT // 2
        ball_dx = random.choice([-1, 1]) * BALL_SPEED
        ball_dy = -BALL_SPEED

        done = False
        rewards_iteration = 0  # Contador de recompensas por iteración

        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    done = True

            state = ball_x // (WIDTH // N_COLS)
            action = agent.get_action(state)

            if action == 0:
                paddle_x -= 20
            elif action == 2:
                paddle_x += 20

            ball_x += ball_dx
            ball_y += ball_dy

            if ball_y + BALL_RADIUS >= HEIGHT - PADDLE_HEIGHT and paddle_x <= ball_x <= paddle_x + PADDLE_WIDTH:
                rewards_iteration += 1

            if ball_x <= BALL_RADIUS or ball_x >= WIDTH - BALL_RADIUS:
                ball_dx *= -1
            if ball_y <= BALL_RADIUS:
                ball_dy *= -1
            if ball_y >= HEIGHT - BALL_RADIUS:
                if paddle_x <= ball_x <= paddle_x + PADDLE_WIDTH:
                    ball_dy *= -1
                else:
                    break

            if paddle_x < 0:
                paddle_x = 0
            if paddle_x > WIDTH - PADDLE_WIDTH:
                paddle_x = WIDTH - PADDLE_WIDTH

            screen.fill(BLACK)
            pygame.draw.rect(screen, WHITE, (paddle_x, HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))
            pygame.draw.circle(screen, WHITE, (ball_x, ball_y), BALL_RADIUS)
            iteration_text = font.render("Iteración: {}".format(iteration), True, WHITE)
            rewards_text = font.render("Recompensas: {}".format(rewards_iteration), True, WHITE)
            max_rewards_text = font.render("Recompensa máxima: {}".format(max_rewards), True, WHITE)

            screen.blit(iteration_text, (10, 10))
            screen.blit(rewards_text, (10, 40))
            screen.blit(max_rewards_text, (10, 70))
            pygame.display.flip()

            next_state = ball_x // (WIDTH // N_COLS)
            reward = 0
            if ball_y >= HEIGHT - BALL_RADIUS:
                reward = -1
            agent.update_q_value(state, action, reward,next_state)

            clock.tick(60)

        iteration += 1
        if rewards_iteration > max_rewards:
            max_rewards = rewards_iteration

    pygame.quit()


if __name__ == "__main__":
    main()