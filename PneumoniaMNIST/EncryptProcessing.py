import logging as log
import torch
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST
import torch.nn as nn
from PneumoniaMNIST.ModelConstruction import PneumoniaMNIST_2taylor
import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import DataLoader
import time
from functools import reduce


root_logger = log.getLogger()
root_logger.setLevel(log.INFO)
handler = log.FileHandler("EncryptionProcessing.log", "w", "utf-8")
handler.setFormatter(log.Formatter(fmt='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S'))
root_logger.addHandler(handler)

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([32,32]),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_set = PneumoniaMNIST(split="test", download=True, size=28, transform=data_transforms,
                          root="/home/cysren/Desktop/ld_file/PPMII/PneumoniaMNIST")

model_file = "/home/cysren/Desktop/ld_file/PPMII/PneumoniaMNIST/PneumoniaMNIST_epoch_35_2taylor.pth"

class taylor_exapnsion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.1992 + torch.multiply(x, 0.5002) + torch.multiply(torch.pow(x, 2), 0.1997)

model = PneumoniaMNIST_2taylor()
model.load_state_dict(torch.load(model_file))
model.eval()
log.info(f"Loading model from file {model_file}")
log.info(model)

# Code for matrix encoding/encryption
# def encode_matrix(HE, matrix):
# #     try:
# #         return np.array(list(map(HE.encodeFrac, matrix)))
# #     except TypeError:
# #         return np.array([encode_matrix(HE, m) for m in matrix])
def encode_matrix(HE, matrix):
    # Encrypt each 1D vector along the last axis (W) of the matrix
    return np.apply_along_axis(HE.encodeFrac, -1, matrix.astype(np.double))

def decode_matrix(HE, matrix):
    # Encrypt each 1D vector along the last axis (W) of the matrix
    return np.apply_along_axis(HE.decodeFrac, -1, matrix.astype(np.double))

def encrypt_matrix(HE, matrix):
    # Encrypt each 1D vector along the last axis (W) of the matrix
    return np.apply_along_axis(HE.encryptFrac, 3, matrix.astype(np.double))

def decrypt_matrix(HE, matrix):
    # Encrypt each 1D vector along the last axis (W) of the matrix
    return np.apply_along_axis(HE.decryptFrac, -1, matrix.astype(np.double))

# def decode_matrix(HE, matrix):
#     try:
#         return np.array(list(map(HE.decodeFrac, matrix)))
#     except TypeError:
#         return np.array([decode_matrix(HE, m) for m in matrix])
#
#
# def encrypt_matrix(HE, matrix):
#     try:
#         return np.array(list(map(HE.encryptFrac, matrix)))
#     except TypeError:
#         return np.array([encrypt_matrix(HE, m) for m in matrix])
#
#
# def decrypt_matrix(HE, matrix):
#     try:
#         return np.array(list(map(HE.decryptFrac, matrix)))
#     except TypeError:
#         return np.array([decrypt_matrix(HE, m) for m in matrix])


# Code for encoded CNN
class ConvolutionalLayer:
    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        result = np.array([[np.sum([convolute2d(image_layer, filter_layer, self.stride)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t])

        if self.bias is not None:
            return np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])
        else:
            return result


def convolute2d(image, filter_matrix, stride):
    x_d = len(image[0])
    y_d = len(image)
    x_f = len(filter_matrix[0])
    y_f = len(filter_matrix)

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(matrix, x, y):
        index_row = y * y_stride
        index_column = x * x_stride
        return matrix[index_row: index_row + y_f, index_column: index_column + x_f]

    return np.array(
        [[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])

def apply_padding(t, padding):
    if isinstance(padding, tuple):
        y_p = padding[0]
        x_p = padding[1]
    else:
        y_p = x_p = padding
    zero = t[0][0][y_p+1][x_p+1] - t[0][0][y_p+1][x_p+1]
    return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]


class LinearLayer:
    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)

    def __call__(self, t):
        result = np.array([[np.sum(image * row) for row in self.weights] for image in t])
        if self.bias is not None:
            result = np.array([row + self.bias for row in result])
        return result

class TaylorExpansionLayer:
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        return taylorExpansion(self.HE, image)


def taylorExpansion(HE, image):
    # Encode the coefficients 0.1550+0.5012x+0.2981x^2-0.0004x^3-0.0388x^4
    x1 = HE.encodeFrac(0.1992)
    x2 = HE.encodeFrac(0.5002)
    x3 = HE.encodeFrac(0.1997)

    try:
        return np.array(list(map(lambda x: HE.add(HE.add_plain(HE.multiply_plain(x, x2), x1),HE.multiply_plain(HE.power(x, 2), x3)), image)))
    except TypeError:
        return np.array([taylorExpansion(HE, m) for m in image])

    # # Result array
    # result = np.empty(image.shape, dtype=object)
    #
    # # Encode each pixel and compute the Taylor expansion
    # it = np.nditer(image, flags=['multi_index', 'refs_ok'], op_flags=['readonly'])
    # while not it.finished:
    #     idx = it.multi_index
    #     x = it[0].item()
    #     # Ensure encrypted_pixel is actually a PyCtxt object before passing to HE.power
    #     if isinstance(x, PyCtxt):
    #         x_pow_2 = HE.power(x, 2)  # Homomorphically compute the square of the pixel value
    #
    #         # result[idx] = HE.add(HE.add_plain(HE.multiply_plain(x, x2), x1), HE.multiply_plain(x_pow_2, x3)),
    #         #                      HE.multiply_plain(x_pow_3, x4)), HE.multiply_plain(x_pow_4, x5))
    #         result[idx] = HE.add(HE.add_plain(HE.multiply_plain(x, x2), x1),HE.multiply_plain(x_pow_2, x3))
    #
    #     else:
    #         print(f"Type error at index {idx}, expected PyCtxt, got {type(x)}")
    #     it.iternext()
    # return result

class BatchNormalLayer:
    def __init__(self, HE, mean, var, epsilon=1e-5, gamma=None, beta=None):
        self.HE = HE
        self.mean = encode_matrix(HE, mean.reshape(1, -1, 1, 1))  # mean is a vector with one value per channel
        self.var = encode_matrix(HE, np.divide(1.0, (np.sqrt(var.reshape(1, -1, 1, 1) + epsilon))))  # sqrt is applied before encoding and encrypting

        if gamma is not None:
            self.gamma = encode_matrix(HE, gamma.reshape(1, -1, 1, 1))
        else:
            self.gamma = None

        if beta is not None:
            self.beta = encode_matrix(HE, beta.reshape(1, -1, 1, 1))
        else:
            self.beta = None

    def __call__(self, t):
        try:
            x_normalized = (t - self.mean) * self.var
            if self.gamma is not None:
                x_normalized = x_normalized * self.gamma
            if self.beta is not None:
                x_normalized = x_normalized + self.beta
        except AttributeError as e:
            print(f"Error during computation: {e}")
            return None
        return x_normalized


class FlattenLayer:
    def __call__(self, image):
        dimension = image.shape
        return image.reshape(dimension[0], dimension[1] * dimension[2] * dimension[3])


class AveragePoolLayer:
    def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        return np.array([[_avg(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])


def _avg(HE, image, kernel_size, stride):
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size[1]
    y_k = kernel_size[0]

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denominator = HE.encodeFrac(1 / (x_k * y_k))

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_column = x * x_s
        return matrix[index_row: index_row + y_k, index_column: index_column + x_k]

    return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]

# We can now define a function to "convert" a PyTorch model to a list of sequential HE-ready-to-be-used layers:
def build_from_pytorch(HE, net):
    # Define builders for every possible layer

    def conv_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()

        return ConvolutionalLayer(HE, weights=layer.weight.detach().numpy(),
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=bias)

    def lin_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
        # type (int, int) or int, unlike in Conv2d
        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size,
                                                                           int) else layer.kernel_size
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        return AveragePoolLayer(HE, kernel_size, stride, padding)

    def flatten_layer(layer):
        return FlattenLayer()

    def taylor_expansion_layer(layer):
        return TaylorExpansionLayer(HE)

    def batch_normal_layer(layer):
        return BatchNormalLayer(HE, layer.running_mean.detach().numpy(),
                                layer.running_var.detach().numpy(), gamma = layer.gamma if hasattr(layer, 'gamma') else None,
                                beta = layer.beta if hasattr(layer, 'beta') else None)

    # Maps every PyTorch layer type to the correct builder
    options = {
        'Conv': conv_layer,
        'Line': lin_layer,
        'AvgP': avg_pool_layer,
        'Flat': flatten_layer,
        'tayl': taylor_expansion_layer,
        'Batc': batch_normal_layer
    }

    # 这里我们确保 net 是 Sequential 或其子模块
    if isinstance(net, nn.Sequential):
        modules = net
    else:
        modules = net.pneumoniaMNIST_2taylor if hasattr(net, 'pneumoniaMNIST_2taylor') else []

    encoded_layers = [options[str(layer)[0:4]](layer) for layer in modules]
    return encoded_layers

log.info(f"Run the experiments...")

n_threads = 8

log.info(f"I will use {n_threads} threads.")

# p = 953983721
# m = 4096

HE = Pyfhel()
ckks_params = {
    "scheme": "CKKS",
    "n": 2**14,
    "scale": 2**40,
    "qi_sizes":[60,40,40,40,60]
}

HE.contextGen(**ckks_params)
HE.keyGen()
HE.rotateKeyGen()
HE.relinKeyGen()

model.to("cpu")
model_encoded = build_from_pytorch(HE, model)

test_loader = DataLoader(test_set, shuffle=True, batch_size=64)


def enc_and_process(image):
    encrypted_image = encrypt_matrix(HE, image.unsqueeze(0).numpy())

    for layer in model_encoded:
        encrypted_image = layer(encrypted_image)

    result = decrypt_matrix(HE, encrypted_image)
    return result


def check_net():
    total_correct = 0
    n_batch = 0

    for batch in test_loader:
        images, labels = batch
        labels = labels.reshape(-1)
        # with parallel_backend('multiprocessing'):
        #     preds = Parallel(n_jobs=n_threads)(delayed(enc_and_process)(image) for image in images)
        for image in images:
            pred = enc_and_process(image)

        preds = reduce(lambda x, y: np.concatenate((x, y)), preds)
        preds = torch.Tensor(preds)

        for image in preds:
            for value in image:
                if value > 100000:
                    log.warning("WARNING: probably you are running out of NB.")

        total_correct += preds.argmax(dim=1).eq(labels).sum().item()
        n_batch = n_batch + 1
        log.info(f"Done {n_batch} batches.")
        log.info(f"This means we processed {n_threads * n_batch} images.")
        log.info(f"Correct images for now: {total_correct}")
        log.info("---------------------------")

    return total_correct


starting_time = time.time()

log.info(f"Start experiment...")

correct = check_net()

end_time = time.time()

total_time = end_time - starting_time
log.info(f"Total corrects on the entire test set: {correct}")
log.info("Time: ", total_time)