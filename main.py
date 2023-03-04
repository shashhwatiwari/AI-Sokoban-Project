import time
from copy import deepcopy as dc
import random
import os
import numpy as np
from time import perf_counter as ptime
from numba import jit, njit, prange
import gc
from heapq import heappop as hpop, heappush as hpsh, heapify as hpfy


# turns string input to integers for numpy
def inter(arr):
    arr = [list(row) for row in arr]
    arr = [[int(char
                .replace(" ", "0")
                .replace("@", "1")
                .replace("+", "2")
                .replace("#", "3")
                .replace("$", "4")
                .replace(".", "5")
                .replace("*", "6"))
            for char in row]
           for row in arr]
    return np.array(arr, dtype=np.uint8)


# checking if the given state has been fully sorted, by checking if there aren't any unplaced boxes (4) left
@jit(forceobj=True, parallel=True)
def solved(sstate):
    return not np.isin(4, sstate)


# pick = f"xsbs/{random.randint(1, 15)}/{random.randint(1, 100)}.xsb"
# while not os.path.exists(pick):
#     pick = f"xsbs/{random.randint(1, 15)}/{random.randint(1, 100)}.xsb"

pick = "newxsbs2/6.xsb"
startstate = open(pick).read().split("\n\n")
lname = pick.split("/")[0] + "/" + startstate[0].split("; ")[1]
startstate = startstate[1].split("\n")
startstate = [list(row) for row in startstate if row]
startstate = inter(startstate)


# we have a list of all 2x2 and 3x3 deadlocks, but it is inefficient to check for *all* of them for each child.
# so we only check the ones that *can* occur in our level by comparing wall structures. waller() strips the level
# and the deadlocks down to just the walls, and sees if latter can even occur here
def waller(mat):
    walled = np.copy(mat)
    for celltype in (1, 2, 4, 5, 6):
        walled[walled == celltype] = 0
    return walled


# checks if smat is in mat. this is to check if there is a deadlock in our state
@jit(forceobj=True, parallel=True)
def submatrix(mat, smat):
    return (np.lib.stride_tricks.sliding_window_view(mat, smat.shape).reshape(-1, *smat.shape) == smat).all(
        axis=(1, 2)).any()


# filtering the deadlocks
allDLs = [[[3, 3], [3, 4]], [[3, 3], [4, 3]], [[4, 3], [3, 3]], [[3, 4], [3, 3]], [[3, 4], [3, 4]], [[3, 3], [4, 4]],
          [[4, 3], [4, 3]], [[4, 4], [3, 3]], [[4, 4], [3, 4]], [[3, 4], [4, 4]], [[4, 3], [4, 4]], [[4, 4], [4, 3]],
          [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[3, 3, 0], [3, 0, 3], [0, 4, 3]],
          [[0, 3, 3], [4, 0, 3], [3, 3, 0]], [[3, 4, 0], [3, 0, 3], [0, 3, 3]], [[0, 3, 3], [3, 0, 4], [3, 3, 0]],
          [[3, 3, 0], [3, 0, 3], [0, 4, 4]], [[0, 3, 3], [4, 0, 3], [4, 3, 0]], [[4, 4, 0], [3, 0, 3], [0, 3, 3]],
          [[0, 3, 4], [3, 0, 4], [3, 3, 0]], [[3, 3, 0], [3, 0, 4], [0, 4, 3]], [[0, 3, 3], [4, 0, 3], [3, 4, 0]],
          [[3, 4, 0], [4, 0, 3], [0, 3, 3]], [[0, 4, 3], [3, 0, 4], [3, 3, 0]], [[3, 3, 0], [3, 0, 4], [0, 4, 4]],
          [[0, 3, 3], [4, 0, 3], [4, 4, 0]], [[4, 4, 0], [4, 0, 3], [0, 3, 3]], [[0, 4, 4], [3, 0, 4], [3, 3, 0]],
          [[3, 4, 0], [3, 0, 4], [0, 4, 4]], [[0, 3, 3], [4, 0, 4], [4, 4, 0]], [[4, 4, 0], [4, 0, 3], [0, 4, 3]],
          [[0, 4, 4], [4, 0, 4], [3, 3, 0]], [[4, 3, 0], [3, 0, 4], [0, 4, 4]], [[0, 3, 4], [4, 0, 3], [4, 4, 0]],
          [[4, 4, 0], [4, 0, 3], [0, 3, 4]], [[0, 4, 4], [3, 0, 4], [4, 3, 0]], [[4, 4, 0], [3, 0, 4], [0, 4, 4]],
          [[0, 3, 4], [4, 0, 4], [4, 4, 0]], [[4, 4, 0], [4, 0, 3], [0, 4, 4]], [[0, 4, 4], [4, 0, 4], [4, 3, 0]],
          [[3, 4, 0], [4, 0, 4], [0, 4, 4]], [[0, 4, 3], [4, 0, 4], [4, 4, 0]], [[4, 4, 0], [4, 0, 4], [0, 4, 3]],
          [[0, 4, 4], [4, 0, 4], [3, 4, 0]], [[4, 4, 0], [4, 0, 4], [0, 4, 4]], [[0, 4, 4], [4, 0, 4], [4, 4, 0]],
          [[4, 4, 0], [4, 0, 4], [0, 4, 4]], [[0, 4, 4], [4, 0, 4], [4, 4, 0]], [[0, 3, 0], [3, 0, 3], [3, 4, 3]],
          [[3, 3, 0], [4, 0, 3], [3, 3, 0]], [[3, 4, 3], [3, 0, 3], [0, 3, 0]], [[0, 3, 3], [3, 0, 4], [0, 3, 3]],
          [[0, 3, 0], [3, 0, 3], [3, 4, 4]], [[3, 3, 0], [4, 0, 3], [4, 3, 0]], [[4, 4, 3], [3, 0, 3], [0, 3, 0]],
          [[0, 3, 4], [3, 0, 4], [0, 3, 3]], [[0, 3, 0], [3, 0, 4], [3, 4, 3]], [[3, 3, 0], [4, 0, 3], [3, 4, 0]],
          [[3, 4, 3], [4, 0, 3], [0, 3, 0]], [[0, 4, 3], [3, 0, 4], [0, 3, 3]], [[0, 3, 0], [3, 0, 4], [3, 4, 4]],
          [[3, 3, 0], [4, 0, 3], [4, 4, 0]], [[4, 4, 3], [4, 0, 3], [0, 3, 0]], [[0, 4, 4], [3, 0, 4], [0, 3, 3]],
          [[0, 3, 0], [3, 0, 3], [4, 4, 4]], [[4, 3, 0], [4, 0, 3], [4, 3, 0]], [[4, 4, 4], [3, 0, 3], [0, 3, 0]],
          [[0, 3, 4], [3, 0, 4], [0, 3, 4]], [[0, 3, 0], [3, 0, 4], [4, 4, 4]], [[4, 3, 0], [4, 0, 3], [4, 4, 0]],
          [[4, 4, 4], [4, 0, 3], [0, 3, 0]], [[0, 4, 4], [3, 0, 4], [0, 3, 4]], [[0, 4, 3], [0, 4, 3]],
          [[0, 0], [4, 4], [3, 3]], [[3, 4, 0], [3, 4, 0]], [[3, 3], [4, 4], [0, 0]], [[0, 3, 0], [4, 0, 4], [4, 4, 4]],
          [[4, 4, 0], [4, 0, 3], [4, 4, 0]], [[4, 4, 4], [4, 0, 4], [0, 3, 0]], [[0, 4, 4], [3, 0, 4], [0, 4, 4]],
          [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 4]], [[4, 4], [4, 6]], [[4, 4], [6, 4]],
          [[6, 4], [4, 4]], [[4, 6], [4, 4]], [[4, 4], [6, 6]], [[6, 4], [6, 4]], [[6, 6], [4, 4]], [[4, 6], [4, 6]],
          [[6, 6], [6, 4]], [[6, 6], [4, 6]], [[4, 6], [6, 6]], [[6, 4], [6, 6]]]
filteredDLs = []
for dl in allDLs:
    if submatrix(waller(startstate), waller(dl)):
        filteredDLs.append(np.array(dl, dtype=np.uint8))


@njit
def ptdist(pt1, pt2):
    return np.abs(pt1[1] - pt2[1]) + np.abs(pt1[0] - pt2[0])


@jit(forceobj=True, parallel=True)
def heur1(mhstate):
    targets = np.argwhere((mhstate == 5))
    boxes = np.argwhere((mhstate == 4))
    sum_ = 0
    for box, target in zip(boxes, targets):
        sum_ += np.abs(box[1] - target[1]) + np.abs(box[0] - target[0])
    return sum_


@jit(forceobj=True, parallel=True)
def heur2(mhstate):
    targets = np.argwhere((mhstate == 5))
    boxes = np.argwhere((mhstate == 4))
    sum_ = 0
    for box in boxes:
        for target in targets:
            sum_ += np.abs(box[1] - target[1]) + np.abs(box[0] - target[0])
    return sum_


@jit(forceobj=True, parallel=True)
def heur3(mhstate):
    return len(np.argwhere((mhstate == 4)))


# checks for possible deadlocks. this function is state-dependent
@jit(forceobj=True, parallel=True)
def deadlock(dlstate):
    for dl in filteredDLs:
        if bool(submatrix(dlstate, dl)):
            return True
    return False


# get player's index
@njit(parallel=True)
def pij(pstate):
    return np.where((pstate == 1) | (pstate == 2))


def wallock(wlstate, ploc, nloc):
    pi, pj = ploc
    bi, bj = nloc
    pi, pj = int(pi), int(pj)
    bi, bj = int(bi), int(bj)

    # up down
    if pi != bi:
        over = bi + 1 if pi < bi else bi - 1
        for cj in prange(bj, 0, -1):
            if wlstate[bi, cj] != 0 or wlstate[over, cj] != 3:
                return False
            if wlstate[bi, cj] == 3:
                break
        for cj in prange(bj, len(wlstate[0])):
            if wlstate[bi, cj] != 0 or wlstate[over, cj] != 3:
                return False
            if wlstate[bi, cj] == 3:
                return True
        return False

    # left right
    elif pj != bj:
        over = bj + 1 if pj < bj else bj - 1
        for ci in prange(bi, 0, -1):
            if wlstate[ci, bj] != 0 or wlstate[ci, over] != 3:
                return False
            if wlstate[ci, bj] == 3:
                break
        for ci in prange(bj, len(wlstate[0])):
            if wlstate[ci, bj] != 0 or wlstate[ci, over] != 3:
                return False
            if wlstate[ci, bj] == 3:
                return True
        return False


# current state's *push* children. tentative: can try backward search with *pull* children as well
@jit(forceobj=True, parallel=True)
def children(pstate):
    pi, pj = pij(pstate)
    ch, cw = pstate.shape
    childs = [pstate]
    for nb1, nb2 in (((pi - 1, pj), (pi - 2, pj)), ((pi + 1, pj), (pi + 2, pj)), ((pi, pj - 1), (pi, pj - 2)),
                     ((pi, pj + 1), (pi, pj + 2))):
        i1, j1 = nb1
        i2, j2 = nb2
        if 0 <= i1 < ch and 0 <= j1 < cw:
            if pstate[i1, j1] in (0, 5):
                cstate = dc(pstate)
                cstate[pi, pj] = 0 if pstate[pi, pj] == 1 else 5
                cstate[i1, j1] = 1 if pstate[i1, j1] == 0 else 2
                childs.append(cstate)
            elif pstate[i1, j1] in (4, 6):
                if 0 <= i2 < ch and 0 <= j2 < cw:
                    if pstate[i2, j2] in (0, 5):
                        cstate = dc(pstate)
                        cstate[pi, pj] = 0 if pstate[pi, pj] == 1 else 5
                        cstate[i1, j1] = 1 if pstate[i1, j1] == 4 else 2
                        cstate[i2, j2] = 4 if pstate[i2, j2] == 0 else 6
                        if not deadlock(cstate) and not wallock(cstate, (pi, pj), nb1):
                            childs.append(cstate)
    return childs


INADM = 100
class State:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        if parent is None:
            self.moves = 0
        else:
            self.moves = parent.moves + 1
        self.heur = heur1(self.state) * INADM
        self.score = self.moves + self.heur

    def __eq__(self, other):
        return (self.state == other.state).all() and self.moves <= other.moves

    def __gt__(self, other):
        return self.heur > other.heur

    def __hash__(self):
        return hash(self.state.tobytes())


print("level dims:", len(startstate), len(startstate[0]))
print("iter moves heur")
stime = ptime()


def solve():
    visited = {}
    notvisited = [State(startstate)]
    hpfy(notvisited)

    i = 0
    while notvisited:
        parent = hpop(notvisited)
        visited.update({parent: parent.moves})

        for child in children(parent.state):
            cnode = State(child, parent)
            if cnode.moves <= 300 and cnode not in visited and cnode not in notvisited:
                hpsh(notvisited, cnode)

        if solved(parent.state):
            print(i, parent.moves)
            route = []
            while parent.parent is not None:
                for row in parent.state:
                    print(row)
                print()
                route.append(parent.state)
                parent = parent.parent
            route.append(startstate)
            route.reverse()
            return i, parent.moves, route

        i += 1
        if int(i) % 1000 == 0:
            print(int(i), int(parent.moves), int(parent.heur / INADM))
            gc.collect()

    print("poopy")  # exhausted depth without finding solution


expansions, route_length, route = solve()
print(ptime() - stime)


indseq = []
for state in route:
    nextind = (np.argwhere((state == 1) | (state == 2))).flatten()
    indseq.append(nextind)

ans = ""
for i in range(len(route) - 1):
    nextind = indseq[i + 1]
    diff = nextind - indseq[i]

    if diff[0] == -1:
        drn = 'u'
    elif diff[0] == 1:
        drn = 'd'
    elif diff[1] == -1:
        drn = 'l'
    else:
        drn = 'r'

    if route[i][nextind[0], nextind[1]] in (4, 6):
        drn = drn.upper()
    ans += drn

print(ans)


import pygame

w = len(startstate[0])
h = len(startstate)

csize = 32
pygame.init()
pygame.display.init()
screen = pygame.display.set_mode(((w + 1) * csize, (h + 1) * csize))
pygame.display.set_caption(lname)

# pygame resources
font = pygame.font.SysFont("Courier", 20, bold=True)

player = pygame.image.load("resources/robot_on_ground.png").convert()
player_on_target = pygame.image.load("resources/robot_on_target.png").convert()
wall = pygame.image.load("resources/wall.png").convert()
box = pygame.image.load("resources/box.png").convert()
empty_target = pygame.image.load("resources/empty_target.png").convert()
full_target = pygame.image.load("resources/full_target.png").convert()
full_ground = pygame.image.load("resources/full_ground.png").convert()
icon = pygame.transform.scale(pygame.image.load("resources/icon.png").convert(), (csize, csize))

print(len(route))
while 1:
    for state in route:
        pygame.event.pump()

        screen.fill((155, 173, 183))
        screen.blit(full_ground, (0, 0))

        for i in range(1, w + 1):
            letter = font.render(chr(i + 64), True, (0, 0, 0))
            rect = pygame.Rect(i * csize, 0, csize, csize)
            pygame.draw.rect(screen, (0, 0, 0), rect, width=2)
            screen.blit(letter, (i * csize + 10, 4))
        for j in range(1, h + 1):
            letter = font.render(chr(j + 96), True, (0, 0, 0))
            rect = pygame.Rect(0, j * csize, csize, csize)
            pygame.draw.rect(screen, (0, 0, 0), rect, width=2)
            screen.blit(letter, (10, j * csize + 4))

        screen.blit(icon, (0, 0))

        pi, pj = 0, 0
        for i in range(w):
            for j in range(h):
                ci = (i + 1) * csize
                cj = (j + 1) * csize
                cell = state[j][i]
                if cell == 1:
                    pi, pj = i, j
                    screen.blit(player, (ci, cj))
                elif cell == 2:
                    pi, pj = i, j
                    screen.blit(player_on_target, (ci, cj))
                elif cell == 3:
                    screen.blit(wall, (ci, cj))
                elif cell == 4:
                    screen.blit(box, (ci, cj))
                elif cell == 5:
                    screen.blit(empty_target, (ci, cj))
                elif cell == 6:
                    screen.blit(full_target, (ci, cj))
        pygame.time.delay(100)
        pygame.display.update()
