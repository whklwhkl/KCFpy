from PIL import Image
import torch
from torchvision import transforms


class VehicleAttributes:

    colors = ['black', 'blue', 'brown', 'cyan', 'dark gray',
              'golden', 'gray', 'green', 'purple', 'red', 'white', 'yellow']

    types = ['small-sized truck', 'HGV/large truck', 'SUV',
             'bulk lorry/fence truck', 'business purpose vehicle/MPV',
             'large-sized bus', 'light passenger vehicle', 'minibus',
             'minivan', 'others', 'pickup truck', 'sedan', 'small-sized truck',
             'tank car/tanker']

    def __init__(self):
        self.model = torch.jit.load('/home/jeff/development/KCFpy/models/attribute_model.zip')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # (h, w)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        print('--- Initialised Vehicle Attributes ---')

    def __call__(self, x):
        # Takes in single image and returns the color and type
        x = Image.fromarray(x)
        x = self.transform(x)
        x.unsqueeze_(0)
        x = x.cuda()
        output = self.model(x)
        idx_color = torch.max(output[0].detach(), 1)[1].cpu().item()
        idx_type = torch.max(output[1].detach(), 1)[1].cpu().item()

        return self.colors[idx_color], self.types[idx_type]