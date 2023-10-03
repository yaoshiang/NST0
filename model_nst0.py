import torch
import torchvision.transforms as transforms
from torchinfo import summary

# model = torchvision.models.vgg11(weights='IMAGENET1K_V1')
# print(model)
# summary(model, (16, 3, 224, 224))


class NST0(torch.nn.Module):

  def __init__(self):
    super(NST0, self).__init__()
    self.image_preprocess = lambda x: (x + 0.5) / 256.
    self.style_preprocess = lambda x: (x + 0.5) / 256.

    self.image_logit_kernel = lambda x: torch.log(x / (1 - x))
    self.style_logit_kernel = lambda x: torch.log(x / (1 - x))

    self.ada_in = lambda image, image_channel_means, image_channel_stds, style_channel_means, style_channel_stds: (
        (image - image_channel_means
        ) / image_channel_stds * style_channel_stds) + style_channel_means

    self.image_sigmoid_kernel = lambda x: torch.sigmoid(x)

    self.image_postprocess = lambda x: torch.round(x * 256. - 0.5)

  def forward(self, image, style):
    # Assert instanceof torch tensor.
    assert isinstance(image, torch.Tensor)
    assert isinstance(style, torch.Tensor)

    image = image.float()
    style = style.float()

    # Assert 3 channel image.
    assert image.dim() >= 3, image.dim
    assert style.dim() >= 3, style.dim

    # Maybe transform [0,1] to [0, 255]
    if torch.max(image) <= 1:
      image = image * 255.
      zero_one = True
    else:
      zero_one = False

    if torch.max(style) <= 1:
      style = style * 255.

    image = self.image_preprocess(image)
    style = self.style_preprocess(style)

    # Strip alpha channel.
    if image.shape[-3] == 4:
      image_alpha = image[..., 3, :, :]
      image = image[..., 0:3, :, :]

    if style.shape[-3] == 4:
      style = style[..., 0:3, :, :]

    # Assert ...CHW format.
    assert image.shape[-3] == 3, image.shape
    assert style.shape[-3] == 3, style.shape

    assert torch.max(image) <= 255
    assert torch.max(style) <= 255

    assert torch.min(image) >= 0
    assert torch.min(style) >= 0

    assert torch.max(image) <= 511 / 512
    assert torch.max(style) <= 511 / 512

    assert torch.min(image) >= 1 / 512
    assert torch.min(style) >= 1 / 512

    image = self.image_logit_kernel(image)
    style = self.style_logit_kernel(style)

    image_channel_means = torch.mean(image, dim=(-2, -1), keepdim=True)
    style_channel_means = torch.mean(style, dim=(-2, -1), keepdim=True)

    image_channel_stds = torch.std(image, dim=(-2, -1), keepdim=True)
    style_channel_stds = torch.std(style, dim=(-2, -1), keepdim=True)

    # AdaIn
    image = self.ada_in(image, image_channel_means, image_channel_stds,
                        style_channel_means, style_channel_stds)

    image = self.image_sigmoid_kernel(image)

    image = self.image_postprocess(image)

    assert torch.max(image) <= 255, torch.max(image)
    assert torch.min(image) >= 0, torch.min(image)

    if zero_one:
      image = image / 255.

    return image
