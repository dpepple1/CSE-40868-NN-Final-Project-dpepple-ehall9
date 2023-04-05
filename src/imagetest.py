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

    def __init__(self):    

        #setup template images
        for filename in os.listdir('Template images'):
            template = cv.imread(f'Template images/{filename}', cv.IMREAD_GRAYSCALE)
            self.templates.append(template)


    def find_enemies(self, obs, i):

        img_rgb = cv.cvtColor(obs, cv.COLOR_BGR2RGB)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

        for key, template in enumerate(self.templates):
            w, h = template.shape[::-1]
            res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
            threshold = 0.6
            loc = np.where( res >= threshold)
            print(key, end = ': ')
            for pt in zip(*loc[::-1]):
                print(f'{i}', f'{pt}', end='  | ')
                cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            print('')