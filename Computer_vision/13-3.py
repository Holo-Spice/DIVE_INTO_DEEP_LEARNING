import torch
import tools.utils as d2l

img = d2l.plt.imread("./1.jpg")
d2l.plt.imshow(img)
d2l.plt.show()

bottle_bbox, blackboard_bbox = [203.0, 197.0, 229.0, 240.0], [100.0, 130.0,362.0, 185.0]

boxes = torch.tensor((bottle_bbox, blackboard_bbox))
print(d2l.box_center_to_corner(d2l.box_corner_to_center(boxes)) == boxes)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(d2l.bbox_to_rect(bottle_bbox,'red'))
fig.axes.add_patch(d2l.bbox_to_rect(blackboard_bbox,'blue'))
d2l.plt.show()