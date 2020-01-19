import cv2, torch, numpy as np

from torchvision.ops import nms
from PIL import Image
from dataclasses import asdict, dataclass
from asyncio import gather


class Detector:
    def __init__(self, model_path, img_size, conf_thres, nms_thres):
        self.model = torch.jit.load(model_path).half().cuda().eval()
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres
        self.img_size = img_size

    async def __call__(self, img:np.ndarray):
        with torch.no_grad():
            img, new_shape = await preprocess(img, self.img_size, torch.float16)
            box, pred = await infer(self.model, img, self.conf_thres, self.nms_thres)
            if len(box):
                box[:, [0, 2]] *= new_shape[1]
                box[:, [1, 3]] *= new_shape[0]
                ret = await gather(*map(postprocess, zip(box, pred)))
                return ret
            else:
                return []


async def preprocess(img:np.array, input_size, dtype):
    h, w, _ = img.shape
    if h * input_size[1] > w * input_size[0]:
        img = np.pad(img, [(0, 0), (0, int(h * input_size[1] / input_size[0]) - w), (0, 0)], mode='constant')
    else:
        img = np.pad(img, [(0, int(w * input_size[0] / input_size[1]) - h), (0, 0), (0, 0)], mode='constant')
    new_shape = img.shape[:-1]
    img = cv2.resize(img, input_size[::-1]).transpose([2, 0, 1]) / 255.  # normalize
    img = torch.from_numpy(img).type(dtype).cuda()
    return img, new_shape


async def infer(model, image:torch.Tensor, confidence_threshold:float, iou_threshold:float):
    """
        image:
            value range from 0 to 1,
            shape [C, H, W]
    """
    with torch.no_grad():
        # outputs range from 0 to 1
        ret = model(image[None])
        box, obj_prob = ret[:2]  # box:(xc, yc, w, h), obj_score
        box_wh = box[:, 2:] / 2
        box_xy = box[:, :2]
        box = torch.cat([box_xy - box_wh, box_xy + box_wh], 1)  # left, top, right, bottom
        keep = torch.where(obj_prob > confidence_threshold)[0]
        obj_prob = obj_prob[keep]
        keep = keep[nms(box[keep], obj_prob, iou_threshold)]
    return box[keep], ret[2][keep] if len(ret) > 2 else obj_prob  # class_conf or obj_score


async def postprocess(args):
    (l, t, r, b), p = args
    class_prob, class_idx = p.max(0)
    class_idx = int(class_idx)
    box = Box(l.item(), t.item(), r.item(), b.item())
    det = Detection(classes[class_idx], class_prob.item(), box)
    return asdict(det)


@dataclass
class Box:
    left:int
    top:int
    right:int
    bottom:int


@dataclass
class Detection:
    label:str
    conf:float
    box:Box


classes = ['person','bicycle','car','motorcycle','airplane','bus','train',
           'truck','boat','traffic light','fire hydrant','stop sign',
           'parking meter','bench','bird','cat','dog','horse','sheep',
           'cow','elephant','bear','zebra','giraffe','backpack','umbrella',
           'handbag','tie','suitcase','frisbee','skis','snowboard',
           'sports ball','kite','baseball bat','baseball glove',
           'skateboard','surfboard','tennis racket','bottle','wine glass',
           'cup','fork','knife','spoon','bowl','banana','apple','sandwich',
           'orange','broccoli','carrot','hot dog','pizza','donut','cake',
           'chair','couch','potted plant','bed','dining table','toilet',
           'tv','laptop','mouse','remote','keyboard','cell phone',
           'microwave','oven','toaster','sink','refrigerator','book',
           'clock','vase','scissors','teddy bear','hair drier','toothbrush']


if __name__ == '__main__':
    detector = Detector('nn_server/half_h416_w416.pt', (416, 416), 0.3, 0.5)

    from asyncio import get_event_loop
    loop = get_event_loop()
    from os.path import expanduser
    img = cv2.imread(expanduser('~/Pictures/bus.jpg'))[...,::-1]
    detection = gather(detector(img),
                       detector(img),
                       detector(img),
                       detector(img),
                       )
    loop.run_until_complete(detection)
    print(detection.result())
