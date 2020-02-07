import os

from time import sleep


VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240


def main():
    print('[ctrl + c] to stop')
    OUTPUT_DIR = '../output'
    while 1:
        with HTML_Diagram_writer('index.html', 'logs') as w:
            # w.add('idx', 'name', header=True)
            for i, d in enumerate(os.listdir(OUTPUT_DIR)):
                # 192.168.66.69_554
                at = os.path.join(OUTPUT_DIR, d)
                if not os.path.isdir(at): continue
                dl = d
                if not os.path.exists(dl):
                    os.mkdir(dl)
                ht = '{}.html'.format(dl)
                # w.add(str(i), w.link(d, ht))
                w.link(d, ht)
                with HTML_Diagram_writer(ht, at,
                                         height=1000,
                                         type='BarChart',
                                         element='Date') as ww:
                    # ww.add('date', header=True)
                    for dd in os.listdir(at):
                        # 2020-01-19
                        att = os.path.join(at, dd)
                        if not os.path.isdir(att): continue
                        ddl = os.path.join(d, dd)
                        if not os.path.exists(ddl):
                            os.mkdir(ddl)
                        htt = '{}.html'.format(ddl)
                        ww.link(dd, htt)
                        # ww.add(ww.link(dd, htt))
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
                                w.add(d)
                                ww.add(dd)
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


class HTML_Diagram_writer:
    def __init__(self, path, title, width=800, height=600, type='PieChart', element='Location'):
        self.name = path
        self.title = title
        self.width = width
        self.height = height
        self.type = type
        self.element = element
        self.file = open(path, 'w')
        self.data = {}
        self.url = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        title = self.field('title', self.title)
        script = self.field('script', '', type='text/javascript',
                            src='https://www.gstatic.com/charts/loader.js')
        jscode = "google.charts.load('current', {'packages':['corechart']});"
        jscode += "google.charts.setOnLoadCallback(drawChart);"
        data = [list(x) for x in self.data.items()]
        data.sort(key=lambda x:-x[1])
        drawChart = '''
function drawChart() {{
    var data = new google.visualization.DataTable();
    data.addColumn('string', '{}');
    data.addColumn('number', 'Counts');
    data.addRows({});
    var options = {{'title':'{}',
                   'width':{},
                   'height':{}}};
    var chart = new google.visualization.{}(document.getElementById('chart_div'));
    var links = {};
    function selectHandler() {{
      var selectedItem = chart.getSelection()[0];
      if (selectedItem) {{
        var topping = data.getValue(selectedItem.row, 0);
        window.location.href = './' + links[topping];
      }}
    }}
    google.visualization.events.addListener(chart, 'select', selectHandler);
    chart.draw(data, options);
}}
        '''.format(self.element, str(data), self.title, self.width, self.height, self.type, str(self.url))
        jscode += drawChart
        # TODO: data and link
        script += self.field('script', jscode, type='text/javascript')
        head = self.field('head', title + script)
        div = self.field('div', '',
                         id='chart_div',
                         style='width:{}; height:{}'.format(self.width,
                                                            self.height))
        body = self.field('body', div)
        html = self.field('html', head + body)
        self.file.write(html)
        self.file.close()

    def add(self, name):
        try:
            self.data[name] += 1
        except KeyError:
            self.data[name] = 1

    def link(self, name, url):
        self.url[name] = url

    @staticmethod
    def field(name, content, **kwargs):
        if len(kwargs):
            start = '<{}>'.format(' '.join([name] + ['{}="{}"'.format(k, v) for k,v in kwargs.items()]))
        else:
            start = '<{}>'.format(name)
        end = '</{}>'.format(name)
        return start + content + end


if __name__ == '__main__':
    main()
