from detection import Detector
from feature import Feature
from attribute import Attribut

from aiohttp import web
from io import BytesIO
from PIL import Image

import numpy as np


routes = web.RouteTableDef()


async def read_image(request):
    reader = await request.multipart()
    field = await reader.next()
    data = await field.read(decode=False)
    img = Image.open(BytesIO(data))
    return img


detector = Detector('half_h416_w416.pt', (416, 416), 0.85, 0.3)
detector_bag = Detector('half_h416_w416_bag.pt', (416, 416), 0.85, 0.3)
feature = Feature('r92m91.pt')
attribute = Attribut('model.pth', .7)


@routes.post('/det')
async def det(request):
    img = await read_image(request)
    img = np.array(img)
    ret = await detector(img)
    return web.json_response(ret)

@routes.post('/det_bag')
async def det(request):
    img = await read_image(request)
    img = np.array(img)
    ret = await detector_bag(img)
    return web.json_response(ret)


@routes.post('/fea')
async def fea(request):
    img = await read_image(request)
    ret = await feature(img)
    return web.json_response(ret)


@routes.post('/att')
async def att(request):
    img = await read_image(request)
    ret = await attribute(img)
    return web.json_response(ret)


if __name__ == '__main__':
    app = web.Application()
    app.router.add_routes(routes)
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--port', default='6666')
    args = ap.parse_args()
    web.run_app(app, port=args.port)
