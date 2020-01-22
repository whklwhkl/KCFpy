import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class Feature:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path).eval().half().cuda()

    async def __call__(self, img):
        img = await preprocess(img)
        with torch.no_grad():
            fea = self.model(img[None].half().cuda())[0].cpu()
        return fea.numpy().tolist()


async def preprocess(img):
    return transform(img)


transform = Compose([
    Resize([256, 128]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    feature = Feature('nn_server/r92m91.pt')
    from asyncio import get_event_loop, gather
    loop = get_event_loop()
    from PIL import Image
    img = torch.zeros(224,224,3, dtype=torch.uint8).numpy()
    img = Image.fromarray(img)
    task = gather(feature(img),
                  feature(img)
                  )
    loop.run_until_complete(task)
    print(task.result())
