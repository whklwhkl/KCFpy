from scipy.spatial.distance import euclidean

import numpy as np
import cv2


def is_unattented(bags, persons, min_dist):
    bc = bags[:, [0, 1]]
    bc += bags[:, [2, 3]]
    bc /= 2
    pc = persons[:, [0, 1]]
    pc += persons[:, [2, 3]]
    pc /= 2
    for b in bc:
        for p in pc:
            if euclidean(b, p) > min_dist:
                return True
    return False


def draw(img, boxes, color=(0,255,0)):
    for l,t,r,b in boxes:
        cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), color, 2)


def test_is_close():
    # setting up
    WIDTH = 800
    HEIGHT = 600
    bag = np.array([[300,400, 400, 500],
                    [500, 500, 550, 550]])  # l,t,r,b
    person = np.array([[300, 100, 350, 300]])
    # animation
    def step(box):
        
        return

    while True:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw(img, bag)
        draw(img, person, (0,0,255))
        # disp
        cv2.imshow('demo', img)
        if cv2.waitKey(1) == 27:
            return



if __name__ == '__main__':
    test_is_close()
