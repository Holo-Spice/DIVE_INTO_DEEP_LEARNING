import torch
import tools.utils as d2l

img = d2l.plt.imread("./1.jpg")
h, w = img.shape[:2]

def display_anchors(fmap_w, fmap_h, s):
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5, 1.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
    d2l.plt.show()



def main():
    print(h, w)
    display_anchors(fmap_w=5, fmap_h=5, s=[0.12])
    display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
    display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

if __name__ == "__main__":
    main()