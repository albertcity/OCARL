import torch
from torch.autograd import Function
import torch.nn as nn

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.shape
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    """
      z_e_x: (n, D)
      latents: (n,) = indices
    """
    def forward(self, z_e_x):
        z_e_x_ = z_e_x
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    """
      z_e_x: (n, D)
      z_q_x_st = z_q_x: (n, D)
    """
    def straight_through(self, z_e_x):
        z_q_x_st, indices = vq_st(z_e_x, self.embedding.weight.detach())

        z_q_x_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x = z_q_x_flatten.view_as(z_e_x)

        return z_q_x_st, z_q_x

#     def forward(self, x):
#         z_e_x = self.encoder(x)
#         z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
#         x_tilde = self.decoder(z_q_x_st)
#         return x_tilde, z_e_x, z_q_x

#         loss_recons = F.mse_loss(x_tilde, images)
#         # Vector quantization objective
#         loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
#         # Commitment objective
#         loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
# 
#         loss = loss_recons + loss_vq + args.beta * loss_commit


