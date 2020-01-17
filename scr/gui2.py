import time
import numpy as np
import cv2

from math import sqrt, ceil


class Main:
    def __init__(self, agents):

        def escape():
            for a in agents:
                a.stop()
            self.root.destroy()

        # self.root.bind('<Escape>', lambda *x: escape())
        self.name = 'main'
        cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # W = self.root.winfo_screenwidth() // 2
        # H = self.root.winfo_screenheight() // 2
        W, H = 1920, 1080
        self.agents = agents
        n = len(agents)
        rows = int(sqrt(n))
        clms = ceil(n/rows)
        self.canvas = np.zeros([H, W, 3]).astype(np.uint8)
        self.panel_size = W // clms, H // rows
        self.y0 = [i // clms * self.panel_size[1] for i in range(n)]
        self.y1 = [i + self.panel_size[1] for i in self.y0]
        self.x0 = [i % clms * self.panel_size[0] for i in range(n)]
        self.x1 = [i + self.panel_size[0] for i in self.x0]
        self.box = list(zip(self.y0, self.y1, self.x0, self.x1))

        def _reset():
            reset = False
            for a in agents:
                a.Track.ALL = set()
                if not reset:
                    a.reset()
                    reset = True

        def _save():
            save = False
            for a in agents:
                a.Track.ALL = set()
                if not save:
                    a.save()
                    save = True

        # self.root.bind('<Delete>', lambda *x: _reset())
        # self.root.bind('<Enter>', lambda *x: _save())

        def toggle_suspend():
            for a in agents:
                a.suspend = not a.suspend

        # self.root.bind('<space>', lambda *x: toggle_suspend())

    def __call__(self):

        while True:
            for i, a in enumerate(self.agents):
                if not a.display_queue.empty():
                    img = a.display_queue.get()
                    img = cv2.resize(img, self.panel_size)
                    y0, y1, x0, x1 = self.box[i]
                    self.canvas[y0:y1, x0:x1] = img
                    cv2.imshow(self.name, self.canvas)
                    if cv2.waitKey(1) == 27:
                        for a in self.agents:
                            a.stop()
                            cv2.destroyAllWindows()
                        return
