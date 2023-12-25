import random
import pygame
class Ant():
    def __init__ (self, row, column):
        self.vision = 1
        self.row = row
        self.column = column
        self.payload = " "
        self.last_position = None
    
    def move(self, boardDimension):
        possibilities = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Remove a posição atual do conjunto de posições visitadas
        if self.last_position:
            possibilities.remove((self.last_position[0] - self.row, self.last_position[1] - self.column))

        # Remove as posições inválidas das possibilidades
        valid_possibilities = [(dr, dc) for dr, dc in possibilities
                              if 0 <= self.row + dr < boardDimension and 0 <= self.column + dc < boardDimension]

        if valid_possibilities:
            direction = random.choice(valid_possibilities)
            next_row = self.row + direction[0]
            next_col = self.column + direction[1]

            # Atualiza a posição anterior
            self.last_position = (self.row, self.column)

            self.row = next_row
            self.column = next_col
    def pintaPos(self, screen, cor, pos):
        pygame.draw.rect(screen, cor, pos)