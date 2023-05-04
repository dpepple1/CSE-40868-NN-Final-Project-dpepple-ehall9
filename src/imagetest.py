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

        self.bolts = cv.imread(f'Template images/bolt.png',cv.IMREAD_GRAYSCALE)
        self.bolts = cv.resize(self.bolts, (0, 0), fx = 0.35, fy = 0.35)

        self.missile = cv.imread(f'Template images/missile.png',cv.IMREAD_GRAYSCALE)
        self.missile = cv.resize(self.bolts, (0, 0), fx = 0.35, fy = 0.35)

    def find_enemies(self, obs):
        im = obs[0:240,0:190]
        im = cv.copyMakeBorder(im,0,0,0,5,cv.BORDER_CONSTANT)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        points = []
        assert im is not None, "file could not be read, check with os.path.exists()"
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv.contourArea(contour) > 0:
                x,y,w,h = cv.boundingRect(contour)
                points.append([x-(w/2),y-(h/2)])
                #cv.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        #cv.imwrite("res.png",im)        
        return points
    
        #img = obs[0:250,0:195]
        #img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        #points = []
        #for key, template in enumerate(self.templates):
        #    w, h = template.shape[::-1]
        #    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
        #    threshold = 0.6
        #    loc = np.where( res >= threshold)
        #    for pt in zip(*loc[::-1]):
        #        #cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        #        points.append(pt)
        #print(points)
        #cv.imwrite("res.png",img_rgb)
        #return points

    def find_self(self, obs):
        img = obs[0:240,0:195]
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        points = []
        w, h = self.ship.shape[::-1]
        res = cv.matchTemplate(img_gray,self.ship,cv.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res>=threshold)
        for pt in zip(*loc[::-1]):
            points.append(pt)
        #cv.rectangle(img_rgb, points[-1], (points[-1][0] + w, points[-1][1] + h), (0,0,255), 2)
        #cv.imwrite("res.png",img_rgb)
        if points:
            return points[-1]
        else:
            return None
    
    def find_bolts(self, obs):
        img = obs[0:240,0:195]
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        points = []
        w, h = self.bolts.shape[::-1]
        res = cv.matchTemplate(img_gray,self.bolts,cv.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res>=threshold)
<<<<<<< HEAD
=======
        for pt in zip(*loc[::-1]):
            points.append(pt)
            #cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        #cv.imwrite("res.png",img_rgb)
        return points
    
    def enemy_missile(self, obs):
        img = obs[0:240,0:195]
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        points = []
        w, h = self.missile.shape[::-1]
        res = cv.matchTemplate(img_gray,self.missile,cv.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res>=threshold)
>>>>>>> b7fc728bd6d106551a561b9ed3598a9bc0f63b5d
        for pt in zip(*loc[::-1]):
            points.append(pt)
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv.imwrite("res.png",img_rgb)
        return points
            
    def fill_grid(self, obs):
        grid = np.zeros((self.xdiv, self.ydiv))
        points = self.find_enemies(obs)
        bolts = self.find_bolts(obs)
        missiles = self.enemy_missile(obs)
        you = self.find_self(obs)
        self.find_self(obs)
        for point in points:
            xmapped = int(point[0] * self.xdiv // self.xres)
            ymapped = int(point[1] * self.ydiv // self.yres)
            grid[xmapped][ymapped] = 1
        for bolt in bolts:
            xmapped = bolt[0] * self.xdiv // self.xres
            ymapped = bolt[1] * self.ydiv // self.yres
            grid[xmapped][ymapped] = 3
<<<<<<< HEAD
        xself = you[0] * self.xdiv // self.xres
        yself = you[1] * self.ydiv // self.yres
        grid[xself][yself] = 2
=======
        for missile in missiles:
            xmapped = missile[0] * self.xdiv // self.xres
            ymapped = missile[1] * self.ydiv // self.yres
            grid[xmapped][ymapped] = 3
        if you:
            xself = you[0] * self.xdiv // self.xres
            yself = you[1] * self.ydiv // self.yres
            grid[xself][yself] = 2
        print(grid)
>>>>>>> b7fc728bd6d106551a561b9ed3598a9bc0f63b5d
        return grid