import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def find_enemy(obs,i):
    img_rgb = cv.cvtColor(obs, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    res = []
    for filename in os.listdir('Template images'):
        template = cv.imread(f'Template images/{filename}', cv.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            print(f'{i}', f'{pt}', end='  | ')
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        print('')
    #cv.imwrite(f'res{i}.png',img_rgb)



class EnemyFinder():

    templates = []

    def __init__(self, xres, yres, xdiv, ydiv):    

        self.xres = xres
        self.yres = yres
        self.xdiv = xdiv
        self.ydiv = ydiv

        #setup template images
        for filename in os.listdir('Template images'):
            template = cv.imread(f'Template images/{filename}', cv.IMREAD_GRAYSCALE)
            self.templates.append(template)

        self.ship = cv.imread(f'Ship Template/ship.png',cv.IMREAD_GRAYSCALE)
        self.ship = cv.resize(self.ship, (0, 0), fx = 0.35, fy = 0.35)


    def find_enemies(self, obs):
        img_rgb = cv.cvtColor(obs, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        points = []

        for key, template in enumerate(self.templates):
            w, h = template.shape[::-1]
            res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
            threshold = 0.6
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                #cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                points.append(pt)
        return points

    def find_self(self, obs):
        img_rgb = cv.cvtColor(obs, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        points = []
        w, h = self.ship.shape[::-1]
        res = cv.matchTemplate(img_gray,self.ship,cv.TM_CCOEFF_NORMED)
        print(res)
        threshold = 0.7
        loc = np.where(res>=threshold)
        for pt in zip(*loc[::-1]):
                cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                points.append(pt)
        cv.imwrite('res.png',img_rgb)
        #return ship

            
    def fill_grid(self, obs):
        grid = np.zeros((self.xdiv, self.ydiv))
        points = self.find_enemies(obs)
        #you = self.find_self(obs)
        self.find_self(obs)
        for point in points:
            xmapped = point[0] * self.xdiv // self.xres
            ymapped = point[1] * self.ydiv // self.yres

            grid[xmapped][ymapped] = 1
        #xself = you[0] * self.xdiv // self.xres
        #yself = you[1] * self.xdiv // self.xres
        #grid[xself][yself] = 2
        return grid