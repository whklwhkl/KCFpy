from flask import Flask, request
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models.densenet import _DenseLayer
from torch.nn import AvgPool2d

import torch, numpy as np
import warnings
warnings.filterwarnings("ignore")


class Attribut:
    def __init__(self, model_path, threshold):
        self.model = torch.load(model_path).eval().cuda()
        for m in self.model.modules():
            if isinstance(m, _DenseLayer):
                m.memory_efficient = False
            elif isinstance(m, AvgPool2d):
                m.divisor_override = None
        self.threshold = threshold

    async def __call__(self, img):
        img = await preprocess(img)
        with torch.no_grad():
            att = self.model(img[None].cuda())
            att = torch.sigmoid(att)[0].cpu()
            att = att[SELECTED_ATTRIBUTES]
        select = att > self.threshold
        att = ATTRIBUTES[att > self.threshold].tolist()
        if not select[0]:
            att = ['Male'] + att
        return att


async def preprocess(img):
    return transform(img)


ATTRIBUTES = np.array(['Female', 'Front', 'Side', 'Back', 'Hat', 'Glasses', 'HandBag',
              'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve',
              'LongSleeve', 'LongCoat', 'Trousers', 'Skirt&Dress'])
SELECTED_ATTRIBUTES = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 24]


transform = Compose([
    Resize((256, 128)),  # (h, w)
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])])


if __name__ == '__main__':
    attribute = Attribut('nn_server/model.pth', .7)
    from asyncio import get_event_loop, gather
    loop = get_event_loop()
    from PIL import Image
    img = torch.zeros(224,224,3, dtype=torch.uint8).numpy()
    img = Image.fromarray(img)
    task = gather(attribute(img),
                  attribute(img)
                  )
    loop.run_until_complete(task)
    print(task.result())
