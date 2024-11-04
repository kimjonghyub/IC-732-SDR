import pygame
import os
import sys

#Initialize the game
pygame.init()

screen=pygame.display.set_mode((640, 480))




#full bar 34 pixel minus 3 on each side so 28
healthvalue=1

#loading images for healthbar
meterbar = pygame.image.load('meter1.png').convert_alpha()
meter = pygame.image.load('meter2.png').convert_alpha()
width = meter.get_width()
height = meter.get_height()
#meter = pygame.Surface((width,height))
run=1
while run:

#Drawing health bar
    screen.fill((255,255,255))
    
    screen.blit(meterbar,(20,20))
    
    for health1 in range(healthvalue):
        
        #meter1 = meter.subsurface((20,20,380,50))
        
        
        screen.blit(meter, (20,20),(0,0,100,50))
        
        #screen.blit(meter1, (20,20))
    #update the screen
    pygame.display.flip()

