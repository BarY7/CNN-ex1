import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        delta_mat = torch.ones(x_scores.shape[0],x_scores.shape[1]) * self.delta
        predicted_mat = y.unsqueeze(1).repeat(1, x_scores.shape[1])
        score_pre_mat = torch.gather(x_scores, 1, predicted_mat)
        minus_mat = torch.sub(x_scores,score_pre_mat)
        final_mat = torch.add(minus_mat,delta_mat)
        #final_mat[final_mat<0] = 0.0
        final_mat = torch.where(final_mat>=0,final_mat,torch.zeros_like(final_mat))
        M = final_mat
        final_mat = torch.sum(final_mat, dim=1)
        temp_delta = torch.ones(final_mat.shape[0])* self.delta
        final_mat = torch.sub(final_mat, temp_delta)
        final_mat = torch.sum(final_mat)
        loss = final_mat * (1/x_scores.shape[0])
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["X_mat"] = x
        self.grad_ctx["M_mat"] = M
        self.grad_ctx["Y_mat"] = y
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.
        x = self.grad_ctx["X_mat"]
        M = self.grad_ctx["M_mat"]
        y = self.grad_ctx["Y_mat"]

        M = torch.where(M <= 0, M, torch.ones_like(M))
        #M[M>0] = 1
        # if torch.all(torch.eq(M,M1)) == False :
        #     print("nor eq")

        y = torch.reshape(y,(y.shape[0],1))
        scatter_mat = torch.zeros(M.shape[0],M.shape[1]).scatter_(1,y,1.0)
        sum_elements = torch.reshape(torch.sum(M, dim=1),(M.shape[0],1)).repeat(1, M.shape[1])
        multi_sum = sum_elements * -1
        final_mat = multi_sum*scatter_mat
        final_mat = (final_mat+M)
        x = torch.transpose(x,0,1)
        grad = torch.mm(x,final_mat)/M.shape[0]

        # ========================

        return grad
