import numpy as np
import torch


class SparseProjectorFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_matrix, projector):
        ctx.input_matrix = input_matrix
        return torch.as_tensor(input_matrix.dot(np.asarray(projector.detach())))

    @staticmethod
    def backward(ctx, grad_output):
        input_matrix = ctx.input_matrix
        grad_input_matrix = grad_projector = None

        if ctx.needs_input_grad[0]:
            raise NotImplementedError()
        if ctx.needs_input_grad[1]:
            grad_projector = torch.as_tensor(input_matrix.transpose().dot(np.asarray(grad_output.detach())))

        return grad_input_matrix, grad_projector


class SparseDPDotFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, left, right, factor):
        ctx.left = left
        ctx.right = right
        return torch.as_tensor(left.kronii_dot(right, np.asarray(factor.detach())))

    @staticmethod
    def backward(ctx, grad_output):
        grad_left = grad_right = grad_factor = None

        if ctx.needs_input_grad[0]:
            raise NotImplementedError()
        if ctx.needs_input_grad[1]:
            raise NotImplementedError()
        if ctx.needs_input_grad[2]:
            grad_factor = torch.as_tensor(ctx.left.kronii_dot_transpose(ctx.right,
                                                                         np.asarray(grad_output.detach())))
        return grad_left, grad_right, grad_factor
