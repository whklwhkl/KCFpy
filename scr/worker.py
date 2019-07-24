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

        self.th = Thread(target=loop, daemon=True)
        self.th.start()

    def put(self, *args):
        self.q.put(args)

    def get(self):
        return self.p.get()
