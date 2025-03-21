import pygame
import numpy as np
from typing import Tuple, Optional
from environment.board import CellType

class Display:
    # Couleurs plus vives et contrastées
    BACKGROUND = (20, 20, 20)  # Fond plus sombre
    GRID_LINE = (50, 50, 50)   # Lignes de grille plus visibles
    SNAKE_HEAD = (0, 255, 0)   # Tête du serpent en vert vif
    SNAKE_BODY = (0, 200, 0)   # Corps du serpent en vert plus foncé
    GREEN_APPLE = (255, 215, 0) # Pomme verte en jaune doré
    RED_APPLE = (255, 0, 0)    # Pomme rouge vif
    TEXT_COLOR = (255, 255, 255) # Texte en blanc

    def __init__(self, board_size: int = 10, cell_size: int = 40):  # Augmentation de la taille des cellules
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        # Ajout d'une marge pour le texte
        self.margin_top = 50
        self.width = board_size * cell_size
        self.height = board_size * cell_size + self.margin_top
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")
        
        # Police plus grande et plus lisible
        self.font = pygame.font.Font(None, 32)
        
        # Step-by-step mode
        self.step_by_step = False
        self.waiting_for_step = False

    def draw_board(self, board: np.ndarray, score: int, max_score: int) -> bool:
        # Remplir l'écran avec la couleur de fond
        self.screen.fill(self.BACKGROUND)
        
        # Dessiner la grille
        for i in range(self.board_size + 1):
            pygame.draw.line(self.screen, self.GRID_LINE, 
                           (i * self.cell_size, self.margin_top),
                           (i * self.cell_size, self.height))
            pygame.draw.line(self.screen, self.GRID_LINE,
                           (0, i * self.cell_size + self.margin_top),
                           (self.width, i * self.cell_size + self.margin_top))
        
        # Dessiner les cellules
        for i in range(self.board_size):
            for j in range(self.board_size):
                cell = board[i][j]
                if cell != CellType.EMPTY.value:
                    color = self.BACKGROUND
                    if cell == CellType.SNAKE_HEAD.value:
                        color = self.SNAKE_HEAD
                    elif cell == CellType.SNAKE_BODY.value:
                        color = self.SNAKE_BODY
                    elif cell == CellType.GREEN_APPLE.value:
                        color = self.GREEN_APPLE
                    elif cell == CellType.RED_APPLE.value:
                        color = self.RED_APPLE
                    
                    # Dessiner avec une petite marge pour voir la grille
                    rect = pygame.Rect(
                        j * self.cell_size + 2,
                        i * self.cell_size + self.margin_top + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, color, rect)
        
        # Afficher les scores en haut
        score_text = self.font.render(f"Length: {score}", True, self.TEXT_COLOR)
        max_score_text = self.font.render(f"Max Length: {max_score}", True, self.TEXT_COLOR)
        
        # Centrer les textes
        score_x = 10
        max_score_x = self.width - max_score_text.get_width() - 10
        text_y = 10
        
        self.screen.blit(score_text, (score_x, text_y))
        self.screen.blit(max_score_text, (max_score_x, text_y))
        
        pygame.display.flip()
        
        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False

    def _draw_cell(self, i: int, j: int, color: Tuple[int, int, int]) -> None:
        """Draw a colored cell at the specified grid position."""
        rect = pygame.Rect(
            j * self.cell_size + 1,
            i * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.screen, color, rect)

    def handle_events(self) -> Optional[bool]:
        """Handle pygame events. Returns True if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.step_by_step:
                    self.waiting_for_step = False
                elif event.key == pygame.K_q:
                    return True
        return False

    def set_step_by_step(self, enabled: bool) -> None:
        """Enable or disable step-by-step mode."""
        self.step_by_step = enabled
        self.waiting_for_step = enabled

    def wait_for_step(self) -> None:
        """Wait for user input in step-by-step mode."""
        if not self.step_by_step:
            return
            
        self.waiting_for_step = True
        while self.waiting_for_step:
            if self.handle_events():
                pygame.quit()
                exit(0)
            pygame.time.wait(50)

    def close(self) -> None:
        """Clean up pygame resources."""
        pygame.quit() 