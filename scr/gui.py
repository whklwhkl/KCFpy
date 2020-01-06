from .drag_n_drop import DragDropRectangle as DDR

import time
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class Map:

    def __init__(self, master, W, H):
        canv = tk.Canvas(master, width=W, height=H)
        canv.grid(row=2, column=2)
        im = Image.open('assets/demo_map_1.jpg').resize([W, H])
        self.tkim = ImageTk.PhotoImage(im)
        canv.create_image(0, 0, anchor='nw', image=self.tkim)
        self.canv = canv
        self.camera_icons = [None] * 3
        self.camera_icons[0] = DDR(canv, 100, 100, 120, 120, 'red')
        self.camera_icons[1] = DDR(canv, 200, 200, 220, 220, 'green')
        self.camera_icons[2] = DDR(canv, 300, 300, 320, 320, 'blue')

    def __call__(self, tracks):
        # find common id in 3 Tracks
        # infer each id's position
        pass

    def _get_coordinate(self, distances):
        pass

    def _get_cam_position(self):
        xys = []
        for cam in self.camera_icons:
            l, t, r, b = self.canv.coords(cam.obj)
            xys += [((l+r)/2, (t+b)/2)]
        return xys


class Main:
    def __init__(self, agents):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)

        def escape():
            for a in agents:
                a.stop()
            self.root.destroy()

        self.root.bind('<Escape>', lambda *x: escape())
        W = self.root.winfo_screenwidth() // 4
        H = self.root.winfo_screenheight() // 2
        self.panel_size = W, H
        self.agents = agents
        self.panels = {}
        self.frames = [None] * 2 * 2

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

        self.root.bind('<Delete>', lambda *x: _reset())
        self.root.bind('<Enter>', lambda *x: _save())

        def toggle_suspend():
            for a in agents:
                a.suspend = not a.suspend

        self.root.bind('<space>', lambda *x: toggle_suspend())
        for i, a in enumerate(agents):
            row = i // 2
            clm = i % 2
            pan = tk.Label(self.root)
            pan.grid(row=1 + row, column=1 + clm)

            def ctrl(event):
                x = event.x / W
                y = event.y / H
                try:
                    self.panels[event.widget].control_queue.put([x,y])
                except KeyError:
                    print('not supported')

            pan.bind('<Button 1>', ctrl)
            self.panels[pan] = a
        # todo: draw the map on the cell 2,2
        if len(agents) == 3:
            self.map_gui = Map(self.root, W, H)

        # todo: add dnd label to represent position of cameras

    def __call__(self):

        def refresh():
            for i, (a, p) in enumerate(zip(self.agents, self.panels)):
                if not a.display_queue.empty():
                    img = a.display_queue.get()
                    im = Image.fromarray(img).resize(self.panel_size)
                    tkim = ImageTk.PhotoImage(im)
                    self.frames[i] = tkim
                    # print(a.frames[i])
                    p.configure(image=self.frames[i])
                    p.update()
            if len(self.agents) == 3:
                # process tracks
                tracks = []
                common_id = set()
                for a in self.agents:
                    track_map = {}
                    for t in a.Track.ALL:
                        if isinstance(t.id, str):
                            track_map[t.id] = t
                            common_id.add(t.id)
                    tracks += [track_map]
                # common_track = {i:[t[i] for t in tracks] for i in common_id}
                common_track = None
                # print(common_track)
                self.map_gui(common_track)
            self.root.after(10, refresh)

        refresh()
        self.root.mainloop()
