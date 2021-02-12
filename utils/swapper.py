# python 3.7
"""Utility functions to invert a given image back to a latent code."""

from tqdm import tqdm
import cv2
import numpy as np

import torch

from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel
from models.interblocks import InterBlocksModel

__all__ = ['StyleGANInverter']


def _softplus(x):
  """Implements the softplus function."""
  return torch.nn.functional.softplus(x, beta=1, threshold=10000)

def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()


class StyleGANSwapper(object):
  """Defines the class for StyleGAN inversion.

  Even having the encoder, the output latent code is not good enough to recover
  the target image satisfyingly. To this end, this class optimize the latent
  code based on gradient descent algorithm. In the optimization process,
  following loss functions will be considered:

  (1) Pixel-wise reconstruction loss. (required)
  (2) Perceptual loss. (optional, but recommended)
  (3) Regularization loss from encoder. (optional, but recommended for in-domain
      inversion)

  NOTE: The encoder can be missing for inversion, in which case the latent code
  will be randomly initialized and the regularization loss will be ignored.
  """

  def __init__(self,
               model_name,
               learning_rate=1e-2,
               iteration=100,
               reconstruction_loss_weight=1.0,
               perceptual_loss_weight=5e-5,
               regularization_loss_weight=2.0,
               logger=None):
    """Initializes the inverter.

    NOTE: Only Adam optimizer is supported in the optimization process.

    Args:
      model_name: Name of the model on which the inverted is based. The model
        should be first registered in `models/model_settings.py`.
      logger: Logger to record the log message.
      learning_rate: Learning rate for optimization. (default: 1e-2)
      iteration: Number of iterations for optimization. (default: 100)
      reconstruction_loss_weight: Weight for reconstruction loss. Should always
        be a positive number. (default: 1.0)
      perceptual_loss_weight: Weight for perceptual loss. 0 disables perceptual
        loss. (default: 5e-5)
      regularization_loss_weight: Weight for regularization loss from encoder.
        This is essential for in-domain inversion. However, this loss will
        automatically ignored if the generative model does not include a valid
        encoder. 0 disables regularization loss. (default: 2.0)
    """
    self.logger = logger
    self.model_name = model_name
    self.gan_type = 'stylegan'

    self.G = StyleGANGenerator(self.model_name, self.logger)
    self.E = StyleGANEncoder(self.model_name, self.logger)
    self.F = PerceptualModel(min_val=self.G.min_val, max_val=self.G.max_val)
    self.S = InterBlocksModel()
    # self.S_BAB = InterBlocksModel()
    self.encode_dim = [self.G.num_layers, self.G.w_space_dim]
    self.run_device = self.G.run_device

    assert list(self.encode_dim) == list(self.E.encode_dim)

    assert self.G.gan_type == self.gan_type
    assert self.E.gan_type == self.gan_type

    self.learning_rate = learning_rate
    self.iteration = iteration
    self.loss_pix_weight = reconstruction_loss_weight
    self.loss_feat_weight = perceptual_loss_weight
    self.loss_reg_weight = regularization_loss_weight
    assert self.loss_pix_weight > 0


  def preprocess(self, image):
    """Preprocesses a single image.

    This function assumes the input numpy array is with shape [height, width,
    channel], channel order `RGB`, and pixel range [0, 255].

    The returned image is with shape [channel, new_height, new_width], where
    `new_height` and `new_width` are specified by the given generative model.
    The channel order of returned image is also specified by the generative
    model. The pixel range is shifted to [min_val, max_val], where `min_val` and
    `max_val` are also specified by the generative model.
    """
    if not isinstance(image, np.ndarray):
      raise ValueError(f'Input image should be with type `numpy.ndarray`!')
    if image.dtype != np.uint8:
      raise ValueError(f'Input image should be with dtype `numpy.uint8`!')

    if image.ndim != 3 or image.shape[2] not in [1, 3]:
      raise ValueError(f'Input should be with shape [height, width, channel], '
                       f'where channel equals to 1 or 3!\n'
                       f'But {image.shape} is received!')
    if image.shape[2] == 1 and self.G.image_channels == 3:
      image = np.tile(image, (1, 1, 3))
    if image.shape[2] != self.G.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{image.shape[2]}, is not supported by the current '
                       f'inverter, which requires {self.G.image_channels} '
                       f'channels!')

    if self.G.image_channels == 3 and self.G.channel_order == 'BGR':
      image = image[:, :, ::-1]
    if image.shape[1:3] != [self.G.resolution, self.G.resolution]:
      image = cv2.resize(image, (self.G.resolution, self.G.resolution))
    image = image.astype(np.float32)
    image = image / 255.0 * (self.G.max_val - self.G.min_val) + self.G.min_val
    image = image.astype(np.float32).transpose(2, 0, 1)

    return image

  def get_init_code(self, image):
    """Gets initial latent codes as the start point for optimization.

    The input image is assumed to have already been preprocessed, meaning to
    have shape [self.G.image_channels, self.G.resolution, self.G.resolution],
    channel order `self.G.channel_order`, and pixel range [self.G.min_val,
    self.G.max_val].
    """
    x = image[np.newaxis]
    x = self.G.to_tensor(x.astype(np.float32))
    z = _get_tensor_value(self.E.net(x).view(1, *self.encode_dim))
    return z.astype(np.float32)

  def swap(self, image_source, image_target,  num_viz=0):
    """Inverts the given image to a latent code.

    Basically, this function is based on gradient descent algorithm.

    Args:
      image_source: Target image to invert, which is assumed to have already been
        preprocessed.
      num_viz: Number of intermediate outputs to visualize. (default: 0)

    Returns:
      A two-element tuple. First one is the inverted code. Second one is a list
        of intermediate results, where first image is the input image, second
        one is the reconstructed result from the initial latent code, remainings
        are from the optimization process every `self.iteration // num_viz`
        steps.
    """
    x_source = image_source[np.newaxis]
    x_target = image_target[np.newaxis]
    x_source = self.G.to_tensor(x_source.astype(np.float32))
    x_target = self.G.to_tensor(x_target.astype(np.float32))
    x_source.requires_grad = False
    x_target.requires_grad = False

    #Get the encoder results
    init_z_source = self.get_init_code(image_source)
    init_z_target = self.get_init_code(image_target)


    #Set up optimizers for the swap blocks
    optimizer = torch.optim.Adam(self.S.parameters(), lr=self.learning_rate)
    #Feed to interblocker
    #Processing source image
    z_AB = self.S.swapAB(init_z_source)
    # x_AB = self.G.net.synthesis(z_AB)

    #Processing target image
    z_BAB = self.S.swapBAB(init_z_target)
    # x_BAB = self.G.net.synthesis(z_BAB)

    viz_results = []


    pbar = tqdm(range(1, self.iteration + 1), leave=True)

    for step in pbar:
      loss = 0.0

      # Reconstruction loss.
      x_source_rec = self.G.net.synthesis(z_AB)
      x_target_rec = self.G.net.synthesis(z_BAB)
      loss_pix_source = torch.mean((x_source - x_source_rec) ** 2)
      loss_pix_target = torch.mean((x_target - x_target_rec) ** 2)

      viz_results.append(self.G.postprocess(_get_tensor_value(x_source_rec))[0])
      viz_results.append(self.G.postprocess(_get_tensor_value(x_target_rec))[0])

      loss = loss + (loss_pix_source + loss_pix_target) * self.loss_pix_weight
      log_message = f'loss_pix: {_get_tensor_value(loss_pix_source + loss_pix_target):.3f}'

      # Perceptual loss.
      if self.loss_feat_weight:
        x_source_feat = self.F.net(x_source)
        x_source_rec_feat = self.F.net(x_source_rec)

        x_target_feat = self.F.net(x_target)
        x_target_rec_feat = self.F.net(x_target_rec)

        loss_source_feat = torch.mean((x_source_feat - x_source_rec_feat) ** 2)
        loss_target_feat = torch.mean((x_target_feat - x_target_rec_feat) ** 2)
        loss = loss + (loss_source_feat + loss_target_feat) * self.loss_feat_weight
        log_message += f', loss_feat: {_get_tensor_value(loss_source_feat + loss_target_feat):.3f}'

      # # Regularization loss.
      # if self.loss_reg_weight:
      #   z_rec = self.E.net(x_rec).view(1, *self.encode_dim)
      #   loss_reg = torch.mean((z - z_rec) ** 2)
      #   loss = loss + loss_reg * self.loss_reg_weight
      #   log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'

      log_message += f', loss: {_get_tensor_value(loss):.3f}'
      pbar.set_description_str(log_message)
      if self.logger:
        self.logger.debug(f'Step: {step:05d}, '
                          f'lr: {self.learning_rate:.2e}, '
                          f'{log_message}')

      # Do optimization.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if num_viz > 0 and step % (self.iteration // num_viz) == 0:
        viz_results.append(self.G.postprocess(_get_tensor_value(x_rec))[0])

    return _get_tensor_value(z), viz_results

  def easy_invert(self, image, num_viz=0):
    """Wraps functions `preprocess()` and `invert()` together."""
    return self.invert(self.preprocess(image), num_viz)

