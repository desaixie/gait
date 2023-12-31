import torch
import socket
import os
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('agg')
from PIL import Image, ImageDraw, ImageFont

from torch.autograd import Variable
import imageio  # saving gif

hostname = socket.gethostname()

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence):
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)
    return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
             hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    assert len(inputs) > 0
    
    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)
        
        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                      (i+1) * x_dim + i * padding, :].copy_(image)
        
        return result
    
    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)
        
        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                         (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    imagetensor = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # convert to uint8, (C, H, W) to (H, W, C)
    return Image.fromarray(imagetensor)

def draw_text_tensor(tensor, text):
    """ assumes CHW tensor, CPU or GPU"""
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)

    # set correct font size
    tensor_size = tensor.size()[-2]
    default_tensor_size = 240
    default_font_size = 20
    new_font_size = int(default_font_size * tensor_size / default_tensor_size)  # scale font size
    # install dejavu: sudo apt-get install -y fonts-dejavu-core
    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"]  # Ubuntu 20.04, 18.04, CentOS 9
    font_path = None
    for p in font_paths:
        if os.path.isfile(p):
            font_path = p
            break
    # font_path = "/usr/share/fonts/truetype/liberation/Liberation/LiberationSans-Regular.ttf"  # Ubuntu 18.04
    font = ImageFont.truetype(font_path, size=new_font_size)
    
    draw.text((4, int(64 * tensor_size / default_tensor_size)), text, (0,0,0), font=font)  # xy, text string, color
    img = np.asarray(pil)
    return torch.Tensor(img / 255.).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        images.append(img.numpy())
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)
