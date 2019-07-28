import tkinter as tk


class DragDrop:
    def __init__(self, widget:tk.Label):
        widget.bind("<ButtonPress-1>", self.on_start)
        widget.bind("<B1-Motion>", self.on_drag)
        widget.bind("<ButtonRelease-1>", self.on_drop)
        widget.configure(cursor="hand1")

    def on_start(self):
        pass

    def on_drag(self):
        pass

    def on_drop(self):
        pass
