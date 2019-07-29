import tkinter as tk


class DragDropRectangle:
    def __init__(self, canvas, left, top, right, bottom, color):
        self.canvas = canvas
        self.xpos, self.ypos = left, top
        self.mouse_xpos = None
        self.mouse_ypos = None
        self.obj = canvas.create_rectangle(left, top, right, bottom, fill=color)

        canvas.tag_bind(self.obj, '<Button1-Motion>', self.move)
        canvas.tag_bind(self.obj, '<ButtonRelease-1>', self.release)
        self.move_flag = False

    def move(self, event):
        if self.move_flag:
            new_xpos, new_ypos = event.x, event.y
            self.canvas.move(self.obj,
                new_xpos-self.mouse_xpos ,new_ypos-self.mouse_ypos)

            self.mouse_xpos = new_xpos
            self.mouse_ypos = new_ypos
        else:
            self.move_flag = True
            self.canvas.tag_raise(self.obj)
            self.mouse_xpos = event.x
            self.mouse_ypos = event.y

    def release(self, event):
        self.move_flag = False
