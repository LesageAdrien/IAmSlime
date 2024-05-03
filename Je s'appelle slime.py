import pygame as pg
import numpy as np
import scipy as sp
CollidingList = []


def surface_tension_sparse_matrix(N):
    M =  np.diag(np.ones(N)) - np.diag(np.ones(N-1), k = -1)
    M[0,-1] =-1
    M += M.T
    return sp.sparse.bsr_matrix(M)

class Slime:
    def __init__(self, n, position = np.array([0,0]), size =200):
        self.n = n
        self.initial_circle = np.zeros((2, n), float)
        self.dots = np.zeros((2, n), float)

        self.angles = np.linspace(0,2*np.pi,n+1)[:-1]
        self.initial_circle[0] = np.cos(self.angles)
        self.initial_circle[1] = np.sin(self.angles)

        self.dots[0] = self.initial_circle[0] * size + position[0]
        self.dots[1] = self.initial_circle[1] * size + position[1]

        self.lastdots = np.copy(self.dots)
        self.lastdots[1] -= 7
        self.center = position
        self.surface = self.computeSurface()
        self.surfacetension = 1e3
        self.surfacefriction = 0.1
        self.airdrag = 1
        self.gravity = 200
        self.size = size
        self.ticktime = 1/30
        self.implicite_matrix = ((1+self.ticktime * self.surfacefriction) * sp.sparse.eye(self.n) + self.ticktime **2 * self.surfacetension * surface_tension_sparse_matrix(self.n))
        self.i = 0
    def disp(self, scr):
        pg.draw.polygon(scr, (50,50,150), self.dots.T)
        pg.draw.circle(scr, (150,50,50), self.center,10)

    def tick(self):
        #Update Moving to each points... By Implicit Method
        forces = np.zeros(self.dots.shape,float)
        #gravity force:
        '''airdrag_coeff = np.maximum(0,(self.dots[0] - self.lastdots[0]) * (self.dots[0] - self.center[0]) + (self.dots[1] - self.lastdots[1]) * (self.dots[1] - self.center[1])) /np.sqrt( (self.dots[0] - self.center[0])**2 + (self.dots[1] - self.center[1])**2) * (self.dots[0] - self.lastdots[0]))'''
        '''airdrag_coeff = ((self.dots[0] - self.lastdots[0]) * (self.dots[0] - self.center[0]) + (self.dots[1] - self.lastdots[1]) * (self.dots[1] - self.center[1]))>0'''
        forces[1] += +self.gravity #- self.airdrag * airdrag_coeff * (self.dots[1] - self.lastdots[1])
        #forces[0]+=  - self.airdrag * airdrag_coeff * (self.dots[0] - self.lastdots[0])/self.ticktime

        #Notice that surface tension forces are implicitly define into self.implicite_matrix
        #Then resolve
        newdots = np.vstack(( sp.sparse.linalg.spsolve(self.implicite_matrix,(2+self.surfacefriction*self.ticktime)*self.dots[0] - self.lastdots[0] + self.ticktime ** 2 * forces[0]), sp.sparse.linalg.spsolve(self.implicite_matrix,(2+self.surfacefriction*self.ticktime)*self.dots[1] - self.lastdots[1] + self.ticktime ** 2 * forces[1])))
        self.lastdots = self.dots
        self.dots = newdots
        #Then update constraint...
        #Angular Constraint:
        self.center = self.computeCenter()
        #norms = np.sqrt((self.dots[0]-self.center[0])**2 + (self.dots[1]-self.center[1])**2)
        norms = np.abs(self.initial_circle[0] * (self.dots[0]-self.center[0]) + self.initial_circle[1] * (self.dots[1]-self.center[1]))
        self.dots[0] = self.initial_circle[0] * (norms )
        self.dots[1] = self.initial_circle[1] * (norms )
        self.size = norms.max()
        #Surface Constraint
        s = (self.dots[0][1:]*self.dots[1][:-1] -  self.dots[1][1:]*self.dots[0][:-1]).sum()/2
        self.dots *= np.sqrt(self.surface/s)
        self.dots[0] += self.center[0]
        self.dots[1] += self.center[1]
    def tick_Collision(self):
        #Collision constraint:
        #just a ground constraint for now
        self.dots[1][self.dots[1]>1000] = 1000
        #Other CollidingBox:
        for collider in CollidingList:
            self.dots = collider.collideWithSlime(slime)
    def computeCenter(self):
        return np.array((self.dots[0].sum(), self.dots[1].sum()))/self.n
    def computeSurface(self):
        tmpx = self.dots[0] - self.center[0]
        tmpy = self.dots[1] - self.center[1]
        return (tmpx[1:]*tmpy[:-1] - tmpy[1:]*tmpx[:-1]).sum()/2
    def getDots(self):
        return self.dots
    def getSize(self):
        return self.size
    def getCorePosition(self):
        return self.center
    def print(self):
        print(self.dots)
    def addSpeed(self, v):
        self.lastdots[0]-=v[0]
        self.lastdots[1]-=v[1]
    def setDots(self, newdots):
        self.dots = newdots
    def collideWithSlime(self, slime, stochastic = 100):
        if np.linalg.norm(slime.getCorePosition() - self.center)< self.size + slime.getSize():
            #do
            #method 1 stochastic update picking, basic, kinda violent
            '''dots = np.copy(slime.getDots())
            vec = self.center - slime.getCorePosition()
            angle = np.arctan2(vec[0],vec[1])/np.pi/2 - 0.25
            for i in range(stochastic):
                v = (self.n*((angle+0.5*np.random.random(2)**3-np.array([-0.25,0.25]))%1)).astype(int)
                det = -self.initial_circle[0,v[0]]*self.initial_circle[1,v[1]] + self.initial_circle[0,v[1]]*self.initial_circle[1,v[0]]
                if det!=0:
                    k1, k2 =  -(- vec[0] * self.initial_circle[1,v[1]] + vec[1] * self.initial_circle[0,v[1]])/det, -(-vec[0] * self.initial_circle[1,v[0]] + vec[1] * self.initial_circle[0,v[0]])/det
                    n1 = np.linalg.norm(self.dots[:,v[0]]-self.center)
                    n2 = np.linalg.norm(dots[:,v[1]]-slime.getCorePosition())
                    if k1>0 and n1>k1 and k2>0 and n2>k2:
                        self.dots[:,v[0]] = self.center+ k1/n1*(self.dots[:,v[0]]- self.center)
                        dots[:,v[1]] = slime.getCorePosition()+ k2/n2*(dots[:,v[1]] -slime.getCorePosition())
            slime.setDots(dots)'''
            #method 2 a for loop updating all of them
            dots = np.copy(slime.getDots())
            vec = self.center - slime.getCorePosition()
            angle = - np.arctan2(vec[0],vec[1])/np.pi/2 + 0.25
            for i in ((int(self.n * angle)+np.arange(-int(self.n/4), int(self.n/4)))%self.n):
                angle0 = -np.arctan2(self.center[0]-dots[0,i], self.center[1]-dots[1,i])/np.pi/2- 0.25
                angle1 = int(self.n*(angle0))%self.n
                ps = (self.initial_circle[0,angle1]*(self.dots[0, angle1]-dots[0,i]) + self.initial_circle[1,angle1]*(self.dots[1, angle1]-dots[1,i]))

                if ps>0:
                    self.dots[:,angle1], dots[:,i] = (self.dots[:,angle1] + dots[:,i])/2, (self.dots[:,angle1] + dots[:,i])/2
            slime.setDots(dots)



class CollisionSphere:
    def __init__(self, radius, position):
        self.radius = radius
        self.position = position
    def getRadius(self):
        return self.radius
    def setRadius(self, radius):
        self.radius = radius
    def getPosition(self):
        return self.position
    def setPosition(self, position):
        self.position = position

    def collideWithSlime(self, slime):
        if np.linalg.norm(slime.getCorePosition()-self.position) < self.radius + slime.getSize():
            #method 1
            '''dots = np.copy(slime.getDots())
            dist = np.sqrt((dots[0] - self.position[0])**2 + (dots[1] - self.position[1])**2)
            mask = dist < (self.radius ** 2)
            dots[0][mask] = self.position[0] + self.radius/dist[mask] * (dots[0][mask]-self.position[0])
            dots[1][mask] = self.position[1] + self.radius/dist[mask] * (dots[1][mask]-self.position[1])
            return dots'''
            #method 2 resolving polynom
            dots = slime.getDots()
            dist = np.sqrt((dots[0] - self.position[0])**2 + (dots[1] - self.position[1])**2)
            mask = dist < (self.radius)

            b = (slime.getCorePosition()[0] -self.position[0])*(dots[0][mask]-slime.getCorePosition()[0]) + (slime.getCorePosition()[1] -self.position[1])*(dots[1][mask]-slime.getCorePosition()[1])

            c = np.square(self.position-slime.getCorePosition()).sum()-self.radius*self.radius

            a = (dots[0, mask] - slime.getCorePosition()[0])**2 + (dots[1, mask] - slime.getCorePosition()[1])**2
            k = (-b-    np.sqrt(b**2-a*c))/a
            dots[0,mask] = slime.getCorePosition()[0]+ (dots[0,mask]- slime.getCorePosition()[0])*k
            dots[1,mask] = slime.getCorePosition()[1]+ (dots[1,mask]- slime.getCorePosition()[1])*k
            return dots
        else:
            return slime.getDots()
    def disp(self, scr):
        pg.draw.circle(scr, (255,0,0), self.position, self.radius)


pg.quit()
pg.init()
SCREENSIZE = np.array(pg.display.get_desktop_sizes()[0])
scr = pg.display.set_mode(SCREENSIZE)
slimes = [Slime(101, position = SCREENSIZE/2),Slime(101, position = SCREENSIZE/2 - 600)]
background = pg.surface.Surface(SCREENSIZE)
background.fill((0,0,0))
pg.draw.line(background, (255,255,255), (0,1000), (1920,1000))
fps = 60
clock = pg.time.Clock()
running = True
dt = 1000
i=0
CollidingList.append(CollisionSphere(100,np.array([0,1000])))
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                print(int(1000/dt))

    scr.blit(background,(0,0))
    for collidebox in CollidingList:
        collidebox.disp(scr)
    if pg.mouse.get_pressed()[1]:
        slimes[0].addSpeed( 0.003*(np.array(pg.mouse.get_pos())- slimes[0].getCorePosition()))
    if pg.mouse.get_pressed()[0]:
        CollidingList[0].setPosition(np.array(pg.mouse.get_pos()))
    if pg.mouse.get_pressed()[2]:
        CollidingList.append(CollisionSphere(50,np.array(pg.mouse.get_pos())))
    for slime in slimes:
        slime.disp(scr)

    dt = clock.tick(fps)
    i+=1
    if i%1 == 0:
        for slime in slimes:
            slime.tick_Collision()
        for i in range(len(slimes)):
            for j in range(i+1,len(slimes)):
                slimes[i].collideWithSlime(slimes[j])
    for slime in slimes:
        slime.tick()
    pg.display.flip()
pg.quit()