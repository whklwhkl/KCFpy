import os

from time import sleep


VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240


def main():
    print('[ctrl + c] to stop')
    OUTPUT_DIR = '../output'
    while 1:
        with HTML_Table_writer('index.html', 'logs') as w:
            w.add('idx', 'name', header=True)
            for i, d in enumerate(os.listdir(OUTPUT_DIR)):
                # 192.168.66.69_554
                at = os.path.join(OUTPUT_DIR, d)
                if not os.path.isdir(at): continue
                dl = d
                if not os.path.exists(dl):
                    os.mkdir(dl)
                ht = '{}.html'.format(dl)
                w.add(str(i), w.link(d, ht))
                with HTML_Table_writer(ht, at) as ww:
                    ww.add('date', header=True)
                    for dd in os.listdir(at):
                        # 2020-01-19
                        att = os.path.join(at, dd)
                        if not os.path.isdir(att): continue
                        ddl = os.path.join(d, dd)
                        if not os.path.exists(ddl):
                            os.mkdir(ddl)
                        htt = '{}.html'.format(ddl)
                        ww.add(ww.link(dd, htt))
                        with HTML_Table_writer(htt, att) as www:
                            www.add('index', 'attributes', 'time', 'video', header=True)
                            for ddd in os.listdir(att):
                                if not ddd.endswith('.avi'): continue
                                # 21_Female_Short Sleeve_Trousers_2020-01-19 18_26_29.475913.avi
                                fields = ddd.split('_')
                                index = fields.pop(0)
                                seconds = fields.pop().split('.')[0]
                                minutes = fields.pop()
                                hours = fields.pop()
                                # the rest is attributes
                                attt = os.path.join(att, ddd)
                                dddl = os.path.join('assets', '{}.mp4'.format(abs(hash(ddd))))
                                if not os.path.exists(dddl):
                                    # os.symlink(attt, dddl)
                                    os.system('ffmpeg -i "{}" "{}"'.format(attt, dddl))
                                www.add(index, '<br>'.join(fields), ':'.join([hours, minutes, seconds]), www.video('/'+dddl))
        try:
            sleep(30)
        except KeyboardInterrupt:
            break


class HTML_Table_writer:
    def __init__(self, path, title):
        self.name = path
        self.file = open(path, 'w')
        title = self.field('title', title)
        meta = ''
        # meta = '''<meta http-equiv="refresh" content="5" >'''
        head = self.field('head', title + meta)
        self.file.write('<html>')
        self.file.write(head)
        self.file.write('<body>')

    def __enter__(self):
        self.file.write('<table style="width:100%">')
        return self

    def __exit__(self, *args):
        self.file.write('</table>')
        self.file.write('</body>')
        self.file.write('</html>')
        self.file.close()

    def add(self, *args, header=False):
        self.file.write('<tr>')
        for a in args:
            s = 'th' if header else 'td'
            self.file.write('<%s>' % s)
            self.file.write(a)
            self.file.write('</%s>' % s)
        self.file.write('/<tr>')

    @staticmethod
    def field(name, content):
        start = '<{}>'.format(name)
        end = '</{}>'.format(name)
        return start + content + end

    @staticmethod
    def video(path):
        return '''
        <video width="{}" height="{}" controls>
            <source src="{}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        '''.format(VIDEO_WIDTH, VIDEO_HEIGHT, path)

    @staticmethod
    def link(name, url):
        return '<a href="{}">{}</a>'.format(url, name)


if __name__ == '__main__':
    main()
