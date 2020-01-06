from queue import Queue
from threading import Thread


class Worker:
    def __init__(self, func):
        self.q = Queue(maxsize=32)  # in
        self.p = Queue(maxsize=32)  # out
        self.running = True

        def loop():
            while self.running:
                try:
                    i = self.q.get()
                    o = func(*i)
                    self.p.put(o)
                except Exception:
                    continue

        self.th = Thread(target=loop)
        self.th.start()

    def has_feedback(self):
        return not self.p.empty()

    def put(self, *args):
        self.q.put(args)

    def get(self):
        return self.p.get()

    def suicide(self):
        self.running = False
        self.th.join(.5)
