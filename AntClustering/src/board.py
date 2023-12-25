import random
import pygame
import sys
from ant import Ant

class Board():
    def __init__(self, dimension, ant_amount, corpse_amount, screen, cell_size, width, height):
        self.dimension = dimension
        self.ants = ant_amount
        self.screen = screen
        self.cell_size = cell_size
        self.window_width = width
        self.window_heigth = height
        self.corpses = corpse_amount

    def board_setting(self):
        if self.corpses > self.dimension**2:
            raise ValueError("ERROR: Too many corpses.")
        
        ant_pop = []
        used_positions = set()

        # Create list of position for ants
        while len(ant_pop) < self.ants:
            row = random.randint(0, self.dimension - 1)
            col = random.randint(0, self.dimension - 1)
            if (row, col) not in used_positions:
                ant_pop.append(Ant(row, col))
                used_positions.add((row, col))

        corpse_board = [[" " for _ in range(self.dimension)] for _ in range(self.dimension)]

        # Generate a list of positions to consider for corpses
        all_positions = [(i, j) for i in range(self.dimension) for j in range(self.dimension)]
        corpses_positions = list(set(all_positions) - used_positions)  # Exclude used ant positions

        # Shuffle corpse positions and put as many as wanted
        random.shuffle(corpses_positions)

        for i in range(self.corpses):
            row, col = corpses_positions[i]
            corpse_board[row][col] = "1"

        return corpse_board, ant_pop

    def cluster(self):
        FLOOR = (255, 255, 255)
        ANT = (255, 0, 0)
        CELL_SIZE = self.cell_size
        game_run = True
        pause = False
        n_int = 0

        base_board = [[False for _ in range(self.dimension)] for _ in range(self.dimension)]
        corpses, ants = self.board_setting()

        while game_run:
            
            n_int = n_int + 1
            print("Num Int: " + str(n_int))
            if n_int >= 1000000:
                pause = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_run = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                pause = True
            if keys[pygame.K_s]:
                pause = False

            if not pause:
                surface = pygame.Surface((self.window_width, self.window_heigth))
                surface.fill(FLOOR)

                # Put corpses on board
                for row in range(self.dimension):
                    for col in range(self.dimension):
                        if corpses[row][col] == "1":
                            base_board[row][col] = True
                            pygame.draw.rect(surface, (0, 0, 0), (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

                # Ant action on board
                for ant in ants:
                    ant.move(self.dimension)
                    row, col = ant.row, ant.column
                    # If there is an object and ant is not carrying (Grab?)
                    if base_board[row][col] and ant.data != "1":
                        grab_seed = 0.5555
                        action_g, ig = ant.action(corpses, self.dimension, "GRAB")
                        if ( grab_seed < action_g):
                            ant.data = corpses[row][col]
                            corpses[row][col] = " "
                            base_board[row][col] = False
                    # Else if there is no object and ant is carrying (Drop?)
                    elif not base_board[row][col] and ant.data == "1":
                        drop_seed = 0.3333
                        action_d, id = ant.action(corpses, self.dimension,  "DROP")
                        if ( drop_seed < action_d ):
                            corpses[row][col] = ant.data
                            ant.data = " "
                            base_board[row][col] = True

                    # Draw Ant
                    pygame.draw.rect(surface, ANT, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))          

                # Update screen
                self.screen.fill(FLOOR)
                self.screen.blit(surface, (0, 0))
                pygame.time.delay(30)
                pygame.display.flip()
            
        pygame.quit()
        sys.exit()

                    
        