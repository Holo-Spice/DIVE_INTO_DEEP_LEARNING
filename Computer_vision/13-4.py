import torch
import tools.utils as d2l

torch.set_printoptions(2) # 精简输出精度

def main():
    img = d2l.plt.imread("./1.jpg")
    h, w = img.shape[:2]
    print(h, w)

    X = torch.rand(size=(1, 3, h, w))
    Y = d2l.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5, 0.1])
    print(Y.shape)

    boxes = Y.reshape(h, w, 6, 4)
    print(boxes[250, 250, 0, :])

    bbox_scale = torch.tensor((w, h, w, h))
    fig = d2l.plt.imshow(img)
    d2l.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                 's=0.75, r=0.5','s=0.75, r=0.1'])
    d2l.plt.show()


if __name__ == '__main__':
    main()