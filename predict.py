import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models



model=models.MyModel()
print(model)

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale=True)])
)



image,target=ds_train[0]

image=image.unsqueeze(dim=0)

model.eval()
with torch.no_grad():
    logits=model(image)

print(logits)

#plt.bar(range(len(logits[0])),logits[0])
#plt.show()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.show(image[0,0],camp='gray_r')
plt.title(f'class:{target}{datasets.FashionMNIST.classes[target]}')

probs=logits.softmax(dim=1)
plt.bar(range(len(probs[0])),probs[0])
plt.ylim(0,1)
plt.title(f'predicted class{probs[0].argmax()}')
plt.show()



