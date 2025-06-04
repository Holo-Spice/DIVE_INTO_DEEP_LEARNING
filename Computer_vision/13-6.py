import os
import pandas as pd
import torch
import torchvision
import tools.utils as d2l
import matplotlib.pyplot as plt

batch_size, edge_size = 32, 256
train_iter, _ = d2l.load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.shown_images(imgs, 2, 5, scale=2)

fig, axes2 = plt.subplots(2, 5, figsize=(5 * 2, 2 * 2))
axes2 = axes2.flatten()

for ax, img in zip(axes2, imgs):
    ax.imshow(img.numpy())
    ax.axis('off')

for ax, label in zip(axes2, batch[1][0:10]):
    coords = (label.squeeze(0)[1:5] * edge_size).detach().numpy()
    d2l.show_bboxes(ax, [torch.tensor(coords)], colors=['w'])

plt.tight_layout()
plt.show()