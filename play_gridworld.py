from gridworld_stuff.gridworld import *
from gridworld_stuff.arch_longwall_2 import *
from gridworld_stuff.gw_draw_utils import *
import time
import pygame

WORLD_WIDTH = arch['WORLD_WIDTH']
WORLD_HEIGHT = arch['WORLD_HEIGHT']
goal = arch['goal']
pits = arch['pits']
walls = arch['walls']
ra_states = arch['ra_states']

def generate_gw():
    gridworld_image = GridWorldImg(WORLD_WIDTH, WORLD_HEIGHT, tile_size=50)
    gridworld_image.agent_loc=(0,0)
    gridworld_image.circle_add(gridworld_image.agent_loc)
    gridworld_image.tile_add((goal[1],goal[0]))
    for w in walls:
        gridworld_image.tile_add((w[1],w[0]), (0, 0, 0))
    for p in ra_states:
        gridworld_image.circle_add((p[1],p[0]), (100, 0, 0),radius_ratio=0.5)
    for p in pits:
        gridworld_image.circle_add((p[1], p[0]), (0, 100, 0))
    return gridworld_image

def change_agent_location(gw_obj,old_loc,new_loc):
    gw_obj.circle_delete(old_loc)
    gw_obj.circle_add(new_loc)
    gw_obj.agent_loc=new_loc

def show_gw(gw_obj):
    gw_obj.update_screen()
    gw_obj.main()

def run_game(gw_obj):
    gw_obj.update_screen()
    while True:
        time.sleep(1)
        move_agent(gw_obj, 'up')
        print('ku')
        gw_obj.update_screen()
    gw_obj.main()

def run_game_iact(gw_obj):
    gw_obj.update_screen()
    i=0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    move_agent(gw_obj, 'up')
                if event.key == pygame.K_DOWN:
                    move_agent(gw_obj, 'down')
                if event.key == pygame.K_RIGHT:
                    move_agent(gw_obj, 'right')
                if event.key == pygame.K_LEFT:
                    move_agent(gw_obj, 'left')
                gw_obj.update_screen()
    gw_obj.main()

def move_agent(gw_obj,direction):
    old_loc=gw.agent_loc
    if direction=='up':
        new_loc = (old_loc[0], old_loc[1]+1)
    if direction=='down':
        new_loc = (old_loc[0], old_loc[1]-1)
    if direction == 'right':
        new_loc = (old_loc[0]+1, old_loc[1])
    if direction == 'left':
        new_loc = (old_loc[0]-1, old_loc[1])
    if new_loc[0]<0:
        return
    if new_loc[0]>(WORLD_WIDTH-1):
        return
    if new_loc[1]<0:
        return
    if new_loc[0]>(WORLD_HEIGHT-1):
        return
    if new_loc in gw_obj.tiles:
        return
    change_agent_location(gw,old_loc,new_loc)

gw=generate_gw()
run_game_iact(gw)
time.sleep(1)
#input('move agent ?')
#change_agent_location(gw,(0,0),(1,1))
#show_gw(gw)


