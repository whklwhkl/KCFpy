import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


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


if __name__ == '__main__':
    root = tk.Tk()
    root.attributes('-fullscreen',True)
    root.bind('<Escape>', lambda *x: root.destroy())
    root.bind('<Button 1>', lambda event: print(event.x, event.y))
    canvas = tk.Canvas(root)
    im = [tk.PhotoImage(Image.open('riQux3XS_400x400.jpg')) for i in range(4)]

    canvas.pack()
    canvas.create_image(0, 0, image=tk.PhotoImage(image=Image.fromarray(np.random.randint(0, 255, [128, 128, 3], np.uint8))), anchor='nw')
    root.mainloop()
