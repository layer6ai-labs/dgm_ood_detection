import math
import torch
from torch import nn
import typing as th
import dypy as dy
from diffusers import UNet2DModel

class BaseNetworkClass(nn.Module):
    def __init__(self, output_split_sizes):
        super().__init__()

        self.output_split_sizes = output_split_sizes

    def _get_correct_nn_output_format(self, nn_output, split_dim):
        if self.output_split_sizes is not None:
            return torch.split(nn_output, self.output_split_sizes, dim=split_dim)
        else:
            return nn_output

    def _apply_spectral_norm(self):
        for module in self.modules():
            if "weight" in module._parameters:
                nn.utils.spectral_norm(module)

    def forward(self, x):
        raise NotImplementedError("Define in child classes.")


class MLP(BaseNetworkClass):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            activation,
            output_split_sizes=None,
            spectral_norm=False
    ):
        super().__init__(output_split_sizes)

        layers = []
        prev_layer_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features=prev_layer_dim, out_features=hidden_dim))
            layers.append(activation())
            prev_layer_dim = hidden_dim

        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        self.net = nn.Sequential(*layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x):
        return self._get_correct_nn_output_format(self.net(x), split_dim=-1)


class BaseCNNClass(BaseNetworkClass):
    def __init__(
            self,
            hidden_channels_list,
            stride,
            kernel_size,
            output_split_sizes=None
    ):
        super().__init__(output_split_sizes)
        self.stride = self._get_stride_or_kernel(stride, hidden_channels_list)
        self.kernel_size = self._get_stride_or_kernel(kernel_size, hidden_channels_list)

    def _get_stride_or_kernel(self, s_or_k, hidden_channels_list):
        if type(s_or_k) not in [list, tuple]:
            return [s_or_k for _ in hidden_channels_list]
        else:
            assert len(s_or_k) == len(hidden_channels_list), \
                "Mismatch between stride/kernels provided and number of hidden channels."
            return s_or_k


class CNN(BaseCNNClass):
    def __init__(
            self,
            input_channels,
            hidden_channels_list,
            output_dim,
            kernel_size,
            stride,
            image_height,
            activation,
            output_split_sizes=None,
            spectral_norm=False,
            noise_dim=0
    ):
        super().__init__(hidden_channels_list, stride, kernel_size, output_split_sizes)

        cnn_layers = []
        prev_channels = input_channels
        for hidden_channels, k, s in zip(hidden_channels_list, self.kernel_size, self.stride):
            cnn_layers.append(nn.Conv2d(prev_channels, hidden_channels, k, s))
            cnn_layers.append(activation())
            prev_channels = hidden_channels

            # NOTE: Assumes square image
            image_height = self._get_new_image_height(image_height, k, s)
        self.cnn_layers = nn.ModuleList(cnn_layers)

        self.fc_layer = nn.Linear(
            in_features=prev_channels*image_height**2+noise_dim,
            out_features=output_dim
        )

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x, eps=None):
        for layer in self.cnn_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)

        net_in = torch.cat((x, eps), dim=1) if eps is not None else x
        net_out = self.fc_layer(net_in)

        return self._get_correct_nn_output_format(net_out, split_dim=1)

    def _get_new_image_height(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        return math.floor((height - kernel)/stride + 1)


class T_CNN(BaseCNNClass):
    def __init__(
            self,
            input_dim,
            hidden_channels_list,
            output_channels,
            kernel_size,
            stride,
            image_height,
            activation,
            output_split_sizes=None,
            spectral_norm=False,
            single_sigma=False,
    ):
        super().__init__(hidden_channels_list, stride, kernel_size, output_split_sizes)

        self.single_sigma = single_sigma

        if self.single_sigma:
            # NOTE: In the MLP above, the single_sigma case can be handled by correctly
            #       specifying output_split_sizes. However, here the first output of the
            #       network will be of image shape, which more strongly contrasts with the
            #       desired sigma output of shape (batch_size, 1). We need the additional
            #       linear layer to project the image-like output to a scalar.
            self.sigma_output_layer = nn.Linear(output_split_sizes[-1]*image_height**2, 1)

        output_paddings = []
        for _, k, s in zip(hidden_channels_list, self.kernel_size[::-1], self.stride[::-1]):
            # First need to infer the appropriate number of outputs for the first linear layer
            image_height, output_padding = self._get_new_image_height_and_output_padding(
                image_height, k, s
            )
            output_paddings.append(output_padding)
        output_paddings = output_paddings[::-1]

        self.fc_layer = nn.Linear(input_dim, hidden_channels_list[0]*image_height**2)
        self.post_fc_shape = (hidden_channels_list[0], image_height, image_height)

        t_cnn_layers = [activation()]
        prev_channels = hidden_channels_list[0]
        for hidden_channels, k, s, op in zip(
            hidden_channels_list[1:], self.kernel_size[:-1], self.stride[:-1], output_paddings[:-1]
        ):
            t_cnn_layers.append(
                nn.ConvTranspose2d(prev_channels, hidden_channels, k, s, output_padding=op)
            )
            t_cnn_layers.append(activation())

            prev_channels = hidden_channels

        t_cnn_layers.append(
            nn.ConvTranspose2d(
                prev_channels, output_channels, self.kernel_size[-1], self.stride[-1],
                output_padding=output_paddings[-1]
            )
        )
        self.t_cnn_layers = nn.ModuleList(t_cnn_layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x):
        x = self.fc_layer(x).reshape(-1, *self.post_fc_shape)

        for layer in self.t_cnn_layers:
            x = layer(x)
        net_output = self._get_correct_nn_output_format(x, split_dim=1)

        if self.single_sigma:
            mu, log_sigma_unprocessed = net_output
            log_sigma = self.sigma_output_layer(log_sigma_unprocessed.flatten(start_dim=1))
            return mu, log_sigma.view(-1, 1, 1, 1)
        else:
            return net_output

    def _get_new_image_height_and_output_padding(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.ConvTranspose2d.html
        # Assume dilation = 1, padding = 0
        output_padding = (height - kernel) % stride
        height = (height - kernel - output_padding) // stride + 1

        return height, output_padding


class GaussianMixtureLSTM(BaseNetworkClass):
    def __init__(self, input_size,  hidden_size, num_layers, k_mixture):
        output_split_sizes = [k_mixture, k_mixture * input_size, k_mixture * input_size]
        super().__init__(output_split_sizes)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=k_mixture*(2*input_size+1))
        self.hidden_size = hidden_size
        self.k = k_mixture

    def forward(self, x, return_h_c=False, h_c=None, not_sampling=True):
        if not_sampling:
            x = torch.flatten(x, start_dim=2)
            x = torch.permute(x, (0, 2, 1))  # move "channels" to last axis
            x = x[:, :-1]  # last coordinate is never used as input
        if h_c is None:
            out, h_c = self.rnn(x)
        else:
            out, h_c = self.rnn(x, h_c)
        if not_sampling:
            out = torch.cat((torch.zeros((out.shape[0], 1, self.hidden_size)).to(out.device), out), dim=1)
        out = self.linear(out)
        weights, mus, sigmas = self.split_transform_and_reshape(out)  # NOTE: weights contains logits
        if return_h_c:
            return weights, mus, sigmas, h_c
        else:
            return weights, mus, sigmas

    def split_transform_and_reshape(self, out):
        weights, mus, sigmas = self._get_correct_nn_output_format(out, split_dim=-1)
        mus = torch.reshape(mus, (mus.shape[0], mus.shape[1], self.k, -1))
        sigmas = torch.reshape(torch.exp(sigmas), (sigmas.shape[0], sigmas.shape[1], self.k, -1))
        return weights, mus, sigmas



"""NOTE: all the U-Net-related classes below (i.e. from Swish to UNet) were taken from
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py"""


class Swish(nn.Module):
    """
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: th.Union[torch.Tensor, type(None)]):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`, can be None if time_channels is 0
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        if self.time_channels > 0:
            # Add time embeddings
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    ### Attention block
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: th.Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: th.Union[torch.Tensor, type(None)]):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool, unet: bool = True):
        super().__init__()

        input_channels = in_channels  # this input is used for resnets
        if unet:
            # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
            # from the first half of the U-Net
            input_channels = in_channels + out_channels
        self.res = ResidualBlock(input_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: th.Union[torch.Tensor, type(None)]):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: th.Union[torch.Tensor, type(None)]):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: th.Union[torch.Tensor, type(None)]):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: th.Union[th.Tuple[int, ...], th.List[int]] = (1, 2, 2, 4),
                 is_attn: th.Union[th.Tuple[bool, ...], th.List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))


class ResidualNetwork(BaseNetworkClass):
    """
    ## Residual network, mainly used to mimic a UNet as much as possible with an encoder/decoder architecture
    """

    def __init__(self,
                 type_: str,
                 input_channels: int,
                 output_channels: int,
                 output_split_sizes: th.Tuple[int, ...],
                 n_channels: int = 256,
                 ch_mults: th.Union[th.Tuple[int, ...], th.List[int]] = (1, 2, 2),
                 n_blocks: int = 2,
                 output_scalar: bool = False,
                 last_conv_size: int = None
                 ):
        """
        * `type_` is either 'encoder' or 'decoder', and specifies if resolutions decrease or increase, respectively
        * `input_channels` is the number of channels in the input. $3$ for RGB encoders.
        * `output_channels` is the number of channels in the final feature map
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `n_blocks` is the number of `ResidualBlocks` at each resolution
        * `output_scalar` only affects encoder, if True applies a final linear layer to have 1-dim outputs
        * `last_conv_size` H*W*C at the final convolution, determines size of the last layer if output_scalar==True
        """
        super().__init__(output_split_sizes)

        assert type_ == 'encoder' or type_ == 'decoder', \
            "type_ for ResidualNetwork has to be `encoder` or `decoder`"
        self.type_ = type_

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        layers = []

        # #### Decreasing resolution
        if type_ == 'encoder':
            # Number of channels
            out_channels = in_channels = n_channels
            # For each resolution
            for i in range(n_resolutions):
                # Number of output channels at this resolution
                out_channels = in_channels * ch_mults[i]
                # Add `n_blocks`
                for _ in range(n_blocks):
                    layers.append(DownBlock(in_channels, out_channels, 0, False))
                    in_channels = out_channels
                # Down sample at all resolutions except the last
                if i < n_resolutions - 1:
                    layers.append(Downsample(in_channels))
        # #### Increasing resolution
        if type_ == 'decoder':
            # Number of channels
            in_channels = out_channels = n_channels
            # For each resolution
            for i in reversed(range(n_resolutions)):
                # `n_blocks` at the same resolution
                out_channels = in_channels
                for _ in range(n_blocks):
                    layers.append(UpBlock(in_channels, out_channels, 0, False, False))
                # Final block to reduce the number of channels
                out_channels = in_channels // ch_mults[i]
                layers.append(UpBlock(in_channels, out_channels, 0, False, False))
                in_channels = out_channels
                # Up sample at all resolutions except last
                if i > 0:
                    layers.append(Upsample(in_channels))

        # Combine the set of modules
        self.layers = nn.ModuleList(layers)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

        # If applicable, apply final linear layer to output scalar
        self.output_linear = None
        if type_ == 'encoder' and output_scalar:
            output_layers = [Swish(), nn.Linear(in_features=last_conv_size, out_features=1)]
            self.output_linear = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """

        # Get image projection
        x = self.image_proj(x)

        for m in self.layers:
            x = m(x, None)

        # Final normalization and convolution
        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        x = self._get_correct_nn_output_format(x, split_dim=1)
        if self.output_linear is not None:
            x = torch.flatten(x, start_dim=1)
            x = self.output_linear(x)
        return x
        # return self.final(self.act(self.norm(x)))

class DiffusersWrapper(nn.Module):
    def __init__(
        self,
        score_network: th.Union[str, torch.nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__()
        
        if isinstance(score_network, str):
            score_network = dy.eval(score_network)
        self.score_network = score_network(*args, **kwargs)
        
    def forward(self, x, t):
        return self.score_network(x, t).sample