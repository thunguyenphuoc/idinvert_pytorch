# python 3.7
"""Contains the encoder class of StyleGAN inversion.

This class is derived from the `BaseEncoder` class defined in `base_encoder.py`.
"""

import numpy as np

import torch


class InterBlocksModel():
  """Defines the encoder class of StyleGAN inversion."""

  def __init__(self, model_name, logger=None):
    self.gan_type = 'stylegan'
    super().__init__(model_name, logger)

  def build(self):
    self.w_space_dim = getattr(self, 'w_space_dim', 512)
    self.encoder_channels_base = getattr(self, 'encoder_channels_base', 64)
    self.encoder_channels_max = getattr(self, 'encoder_channels_max', 1024)
    self.use_wscale = getattr(self, 'use_wscale', False)
    self.use_bn = getattr(self, 'use_bn', False)
    self.ABNet = SwapNetAB(
        in_channels=self.image_channels,
        out_channels=self.encoder_channels_base)

    self.BNet = SwapNetB(
        in_channels=self.image_channels,
        out_channels=self.encoder_channels_base)
    self.num_layers = self.net.num_layers
    self.encode_dim = [self.num_layers, self.w_space_dim]

  def _swapAB(self, encoded_z):
    encoded_z = self.to_tensor(encoded_z.astype(np.float32))
    codes = self.ABnet(encoded_z)
    concat_codes = torch.cat((codes, codes), 0)

    assert concat_codes.shape == (encoded_z.shape[0], np.prod(self.encode_dim))
    concat_codes = concat_codes.view(encoded_z.shape[0], *self.encode_dim)
    results = {
        'encoded_z': encoded_z,
        'code': self.get_value(concat_codes),
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def swapAB(self, encoded_zs, **kwargs):
    return self.batch_run(encoded_zs, self._swapAB)

  def _swapBAB(self, encoded_z):
    encoded_z = self.to_tensor(encoded_z.astype(np.float32))
    codeAB = self.ABnet(encoded_z)
    codeB = self.Bnet(encoded_z)
    concat_codes = torch.cat((codeB, codeAB), 0)

    assert concat_codes.shape == (encoded_z.shape[0], np.prod(self.encode_dim))
    concat_codes = concat_codes.view(encoded_z.shape[0], *self.encode_dim)
    results = {
        'encoded_z': encoded_z,
        'code': self.get_value(concat_codes),
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def swapBAB(self, encoded_zs, **kwargs):
    return self.batch_run(encoded_zs, self._swapBAB)




#Defining actual MLPs
class SwapNetAB(nn.Module):
  """Implements the last block, which is a dense block."""

  def __init__(self,
               in_channels,
               out_channels,
               ):
    super().__init__()

    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=False)
    self.fc1 = nn.Linear(in_features=out_channels,
                        out_features=out_channels,
                        bias=False)



  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    x = self.fc1(x)
    # x = x.view(x.shape[0], x.shape[1], 1, 1)
    # return self.bn(x).view(x.shape[0], x.shape[1])
    return x


class SwapNetB(nn.Module):
  """Implements the last block, which is a dense block."""

  def __init__(self,
               in_channels,
               out_channels,
               ):
    super().__init__()

    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=False)
    self.fc1 = nn.Linear(in_features=out_channels,
                        out_features=out_channels,
                        bias=False)



  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    x = self.fc1(x)
    # x = x.view(x.shape[0], x.shape[1], 1, 1)
    # return self.bn(x).view(x.shape[0], x.shape[1])
    return x

