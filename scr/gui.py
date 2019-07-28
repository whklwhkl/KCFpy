import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from glob import glob


class PoppinParty:
    def __init__(self):
        self.window = tk.Tk()
        self.label = tk.Label(self.window, text='name:')
        self.label.pack()
        self.entry = tk.Entry(self.window)
        self.entry.pack()
        self.content = None

        def set_name():
            self.content = self.entry.get()
            self.destroy()

        self.buttonY = tk.Button(self.window, text='confirm', command=set_name)
        self.buttonY.pack()
        self.buttonN = tk.Button(self.window, text='cancle', command=self.destroy)
        self.buttonN.pack()

    def show(self):
        target = self.window.mainloop()

    def destroy(self):
        self.window.destroy()


class Main:
    def __init__(self, agents):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)

        def escape():
            for a in agents:
                a.stop()
            self.root.destroy()

        self.root.bind('<Escape>', lambda *x: escape())
        W = self.root.winfo_screenwidth() // 2
        H = self.root.winfo_screenheight() // 2
        self.panel_size = W, H
        self.agents = agents
        self.panels = {}
        self.frames = [None] * 2 * 2
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
            self.root.after(20, refresh)

        refresh()
        self.root.mainloop()
