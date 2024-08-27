import math
from time import time

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import trange


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2, kernel_size[1]//2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                out_channels=4*self.hidden_dim,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED/blob/master/utils/matrix_generator.py
    
def generate_mscred_signature_matrices(data_in,
                                        window_sizes=[10, 30, 60],
                                        step=10,
                                        device='cuda:0'):

    data_in.to(device)
    length, channels = data_in.shape
    data_out = []

	# Generate multi-scale signature matrices

    for t in trange(max(window_sizes), length, step,
                        desc='Generating MSCRED signature matrices'):

        # Prepare the tensor to store results
        element = torch.empty((channels,
                                    channels,
                                    len(window_sizes)),
                                device=device)

        # Using torch.einsum to compute the inner products
        for c, window_size in enumerate(window_sizes):
            sliced = data_in[t - window_size:t, :]

            result = torch.einsum('ki,kl->li', sliced, sliced)

            element[:, :, c] = result
        
        data_out.append(element)

        assert(torch.all(torch.isclose(element, element)))

    data_out = torch.stack(data_out)

    scale_factor = torch.from_numpy(np.array([[[window_sizes]]],
                                                    dtype=np.float64))

    scale_factor = scale_factor.detach().to(device)

    data_out = data_out/scale_factor

    # print(data_out.shape)

    data_out = torch.permute(data_out, (0, 3, 1, 2))

    # print(data_out.shape)

    return data_out


# def generate_mscred_signature_matrices(data_in,
#                                         window_sizes=[10, 30, 60],
#                                         step=10,
#                                         block_size=5,
#                                         device='cuda:0'):

#     data_in.to(device)
#     length, channels = data_in.shape

#     element = None
#     data_out_block = []
#     data_out_all = []

# 	# Generate multi-scale signature matrices

#     for t in trange(max(window_sizes), length, step,
#                         desc='Generating MSCRED signature matrices'):

#         # Prepare the tensor to store results
#         element = torch.empty((channels,
#                                     channels,
#                                     len(window_sizes)),
#                                 device=device)

#         # Using torch.einsum to compute the inner products
#         for c, window_size in enumerate(window_sizes):
#             sliced = data_in[t - window_size:t, :]

#             result = torch.einsum('ki,kl->li', sliced, sliced)

#             element[:, :, c] = result
        
#         data_out_block.append(element)

#         if len(data_out_block) >= block_size:
#             data_out_all.append(data_out_block)
#             data_out_block = []

#     while len(data_out_all[-1]) < 5:
#         data_out_all[-1].append(element)

#     data_out = [torch.stack(block) for block in data_out_all]

#     data_out = torch.stack(data_out)

#     scale_factor = torch.from_numpy(np.array([[[[window_sizes]]]],
#                                                     dtype=np.float64))

#     scale_factor = scale_factor.detach().to(device)

#     data_out = data_out/scale_factor

#     print(data_out.shape)

#     data_out = torch.permute(data_out, (0, 1, 4, 2, 3))

#     print(data_out.shape)

#     return data_out


# https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED/blob/master/model/mscred.py

class MSCREDBaseConvLSTMCell(nn.Module):
    def __init__(self,
                    input_channels,
                    hidden_channels,
                    kernel_size,
                    device='cuda:0'):
        super(MSCREDBaseConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.device = device
        self.num_features = 4

        self.padding = int((kernel_size - 1)/2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c*self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c*self.Wcf)
        cc = cf*c + ci*torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc*self.Wco)
        ch = co*torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros((1, hidden,
                                                    shape[0],
                                                    shape[1]),
                                                device=torch.device(self.device),
                                                dtype=torch.float64))
            self.Wcf = Variable(torch.zeros((1, hidden,
                                                    shape[0],
                                                    shape[1]),
                                                    device=torch.device(self.device),
                                                dtype=torch.float64))
            self.Wco = Variable(torch.zeros((1, hidden,
                                                    shape[0],
                                                    shape[1]),
                                                device=torch.device(self.device),
                                                dtype=torch.float64))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

        return (Variable(torch.zeros((batch_size,
                                                hidden,
                                                shape[0],
                                                shape[1]),
                                            device=torch.device(self.device),
                                            dtype=torch.float64)),
                Variable(torch.zeros((batch_size,
                                                hidden,
                                                shape[0],
                                                shape[1]),
                                            device=torch.device(self.device),
                                            dtype=torch.float64)))


class MSCREDBaseConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(MSCREDBaseConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = MSCREDBaseConvLSTMCell(self.input_channels[i],
                                            self.hidden_channels[i],
                                            self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):

        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(
                                                    batch_size=bsize,
                                                    hidden=self.hidden_channels[i],
                                                    shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            # if step in self.effective_step:
            outputs.append(x)

        outputs = torch.stack(outputs, dim=1)

        return outputs, (x, new_c)


def mscred_attention(conv_lstm_out):

    # print(conv_lstm_out.shape)

    attention_w = []
    for k in range(5):
        attention_w.append(torch.sum(torch.mul(conv_lstm_out[:, k], conv_lstm_out[:, -1]))/5)
    m = nn.Softmax()

    attention_w = torch.reshape(m(torch.stack(attention_w)), (-1, 5))

    cl_out_shape = conv_lstm_out.shape

    conv_lstm_out = torch.reshape(conv_lstm_out, (5, -1))
    conv_lstm_out = torch.matmul(attention_w, conv_lstm_out)

    conv_lstm_out = torch.reshape(conv_lstm_out, (cl_out_shape[0],
                                                    cl_out_shape[2],
                                                    cl_out_shape[3],
                                                    cl_out_shape[4]))

    return conv_lstm_out


class MSCREDEncoder(nn.Module):
    def __init__(self, in_channels_encoder):
        super(MSCREDEncoder, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1),
                        nn.SELU())
        
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, (2, 2), 1),
                        nn.SELU())
          
        self.conv3 = nn.Sequential(
                        nn.Conv2d(64, 128, 2, (2, 2), 1),
                        nn.SELU())
         
        self.conv4 = nn.Sequential(
                        nn.Conv2d(128, 256, 2, (2, 2), 0),
                        nn.SELU())
        
    def forward(self, X):
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out


class MSCREDConvLSTM(nn.Module):
    def __init__(self):
        super(MSCREDConvLSTM, self).__init__()

        self.conv1_lstm = MSCREDBaseConvLSTM(input_channels=32, hidden_channels=[32], 
                                                kernel_size=3, step=5, effective_step=[4])
        self.conv2_lstm = MSCREDBaseConvLSTM(input_channels=64, hidden_channels=[64], 
                                                kernel_size=3, step=5, effective_step=[4])
        self.conv3_lstm = MSCREDBaseConvLSTM(input_channels=128, hidden_channels=[128], 
                                                kernel_size=3, step=5, effective_step=[4])
        self.conv4_lstm = MSCREDBaseConvLSTM(input_channels=256, hidden_channels=[256], 
                                                kernel_size=3, step=5, effective_step=[4])

    def forward(self, conv1_out, conv2_out, 
                        conv3_out, conv4_out):

        # conv1_out = F.pad(conv1_out,
        #                     (1, 2, 1, 2),
        #                     mode='replicate')

        conv1_lstm_out = self.conv1_lstm(conv1_out)
        conv1_lstm_out = mscred_attention(conv1_lstm_out[0])
        conv2_lstm_out = self.conv2_lstm(conv2_out)
        conv2_lstm_out = mscred_attention(conv2_lstm_out[0])
        conv3_lstm_out = self.conv3_lstm(conv3_out)
        conv3_lstm_out = mscred_attention(conv3_lstm_out[0])
        conv4_lstm_out = self.conv4_lstm(conv4_out)
        conv4_lstm_out = mscred_attention(conv4_lstm_out[0])

        return conv1_lstm_out,\
                conv2_lstm_out,\
                conv3_lstm_out,\
                conv4_lstm_out

class MSCREDDecoder(nn.Module):
    def __init__(self, in_channels):
        super(MSCREDDecoder, self).__init__()

        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0),
                        nn.SELU())
        
        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 2, 2, 1, 1),
                        nn.SELU())
        
        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(128, 32, 3, 2, 1, 1),
                        nn.SELU())
        
        self.deconv1 = nn.Sequential(
                            nn.ConvTranspose2d(64, 3, 3, 1, 1, 0),
                            nn.SELU())
    
    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        deconv4 = self.deconv4(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim=1)
        deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim=1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim=1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1


class PositionalEncoding(nn.Module):
    def __init__(self,
                    d_model,
                    dropout=0.1,
                    max_len=5000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float()*(-math.log(10000.0) / d_model))
        pe += torch.sin(position*div_term)
        pe += torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)
        
    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True) #nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True) #nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
