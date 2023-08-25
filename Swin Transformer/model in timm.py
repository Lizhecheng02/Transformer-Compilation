import torch
import torchvision
import timm
import json
from urllib.request import urlopen
from PIL import Image
from torchvision import transforms

URL = 'https://raw.githubusercontent.com/SharanSMenon/swin-transformer-hub/main/imagenet_labels.json'

model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
])

image = Image.open('./Data/test.jpg')
transformed = transform(image)
batch = transformed.unsqueeze(0)

with torch.no_grad():
    output = model(batch)

class_idx = output.argmax(dim=1)
print(class_idx)

response = urlopen(URL)
classes = json.loads(response.read())
print(len(classes))

print(classes[class_idx])
