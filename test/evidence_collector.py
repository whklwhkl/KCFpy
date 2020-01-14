from random import random
from time import sleep


def frames():
    while True:
        yield random()


class Detective:
    def __init__(self, length=10):
        self.evidence = []
        self.length = length
        self.idx = 0

    def __call__(self, new_evidence):
        result = '?'
        if len(self.evidence) < self.length:
            self.evidence.append(new_evidence)
        else:
            # self.evidence[self.idx % self.length] = new_evidence
            # result = sum(self.evidence) > 5
            result = sum(self.evidence) > 5
            self.evidence = [new_evidence]
        self.idx += 1
        return result


def main():
    d = Detective()
    for e in frames():
        r = d(e)
        print(e, '=>', r)
        sleep(1)


if __name__ == '__main__':
    main()
