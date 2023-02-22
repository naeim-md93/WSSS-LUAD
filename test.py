import torch
import torch.nn.functional as F

x = torch.randn(1, 172, 220, 156)
kc, kh, kw = 32, 32, 32  # kernel size
dc, dh, dw = 32, 32, 32  # stride
# Pad to multiples of 32
x = F.pad(x, (x.size(2)%kw // 2, x.size(2)%kw // 2,
              x.size(1)%kh // 2, x.size(1)%kh // 2,
              x.size(0)%kc // 2, x.size(0)%kc // 2))

patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
unfold_shape = patches.size()
patches = patches.contiguous().view(-1, kc, kh, kw)
print(patches.shape)

# Reshape back
patches_orig = patches.view(unfold_shape)
output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]
patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.view(1, output_c, output_h, output_w)

# Check for equality
print((patches_orig == x[:, :output_c, :output_h, :output_w]).all() == torch.tensor(1, dtype=torch.uint8))

from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from src.models.resnet38d_cls import create_patches

img = Image.open(fp='test/data/1031280-4092-15376-[1 0 0 1].png')
img = T.ToTensor()(img).unsqueeze(0)
print(img.size())
plt.imshow(img[0].moveaxis(0, -1))
plt.show()

patches = create_patches(x=img, kernel_size=28, stride=14)

# kc, kh, kw = 3, 112, 112  # kernel size
# dc, dh, dw = 3, 56, 56  # stride
#
# img = F.pad(img, (img.size(2)%kw // 2, img.size(2)%kw // 2,
#               img.size(1)%kh // 2, img.size(1)%kh // 2,
#               img.size(0)%kc // 2, img.size(0)%kc // 2))
#
# patches = img.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
# unfold_shape = patches.size()
# patches = patches.contiguous().view(-1, kc, kh, kw)
# print(patches.shape)
#
# # Reshape back
# patches_orig = patches.view(unfold_shape)
# output_c = unfold_shape[1] * unfold_shape[4]
# output_h = unfold_shape[2] * unfold_shape[5]
# output_w = unfold_shape[3] * unfold_shape[6]
# patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
# patches_orig = patches_orig.view(1, output_c, output_h, output_w)

# Check for equality
# print((patches_orig == img[:, :output_c, :output_h, :output_w]).all() == torch.tensor(1, dtype=torch.uint8))
# for x in range(patches.size(0)):
#     plt.imshow(patches[x].moveaxis(0, -1))
#     plt.show()
print(patches.size())