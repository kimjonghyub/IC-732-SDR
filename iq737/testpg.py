import pygame
import time
import os
pygame.init()

class bar:
    def __init__(self, maximum=100, minimum=0, ticklen=2, height=100, bgcolor=[0,0,0], bcolor=[0,255,0], wx=None, wy=None): #maximum value, minimum value, px between whole-number locations on bar, height in pixels of bar window, background color, bar color
        if wx != None and wy != None:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (int(wx),int(wy))

        if maximum <= minimum:
            raise SyntaxError('Minimum >= Maximum')
        
        self.mx = maximum
        self.mn = minimum
        self.pos = 0
        self.scl = ticklen
        self.h = height
        self.area = [2, 2, 0, self.h - 4]
        self.bc = bcolor
        self.bgc = bgcolor
        self.screen = pygame.display.set_mode([self.scl * (maximum - minimum) + 4, height], pygame.NOFRAME)
        self.screen.fill(self.bgc)
        pygame.display.flip()
    def setval(self, val): #sets bar length to val
        if val <= self.mx and val >= self.mn:
            self.area[2] = val * self.scl
            self.screen.fill(self.bgc)
            barsurf = pygame.Surface([self.area[2], self.area[3]])
            barsurf.fill(self.bc)
            self.screen.blit(barsurf, self.area[:2])
            
            self.pos = val
            pygame.display.flip()
        else:
            raise IndexError('Val > Maximum or < Minimum')

    def end(self): #closes the bar
        pygame.display.quit()
