#from torch.utils.data import Dataset
from typing import Union, List, Optional
from torch.nn import Sequential, Module, Linear, Softmax, Flatten
import torch 
import numpy as np

# Adapted from https://github.com/IBM/simulai
# (simulai/models/_pytorch_models/_miscellaneous.py:MoEPool)
# also under license Apache 2.0. 

class MoELayer(Module):
    def __init__(
        self,
        experts_list: List[Module],
        gating_network: callable = None,
        input_size: int = None,
        n_vars : int = 2, 
        k: int = 3,
        alpha: float = 0.2,
        devices: Union[list, str] = None,
        binary_selection: bool = False,
        load_balancing: bool = False,
        hidden_size: Optional[int] = None,
        use_reshaping: bool = False
    ) -> None:
        """A layer to execute Mixture of Experts

        Args:
            experts_list (List[None]): The list of neural networks used as experts.
            gating_network (Union[callable], optional): Network or callable operation used for predicting
        weights associated to the experts. (Default value = None)
            input_size (int, optional): The number of dimensions of the input. (Default value = None)
            devices (Union[list, str], optional): Device ("gpu" or "cpu") or list of devices in which
        the model is placed. (Default value = None)
            binary_selection (bool, optional): The weights will be forced to be binary or not. (Default value = False)
            hidden_size (Optional[int], optional): If information about the experts hidden size is required, which occurs,
        for instance, when they are ConvexDenseNetwork objects,
        it is necessary to define this argument. (Default value = None)

        """

        super(MoELayer, self).__init__()

        self.n_experts = len(experts_list)
        self.input_size = input_size
        self.n_vars = n_vars
        self.hidden_size = hidden_size
        self.binary_selection = binary_selection
        self.is_gating_trainable = True
        self.k = k 
        self.alpha = alpha 
        self.use_reshaping = use_reshaping

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Gating (classifier) network/object
        # The default gating network is a single-layer fully-connected network
        if gating_network is None:
            gating_network = Sequential(
                Linear(self.input_size * self.n_vars, self.n_experts, bias=False),
                Softmax(),
            )

            self.gating_network = gating_network

        else:

            self.gating_network = gating_network

        self.gating_network.to(self.device)

        self.experts_list = []
        # Sending each sub-network to the correct device
        for expert in experts_list:
            self.experts_list.append(expert.to(self.device))

        self.experts_list = torch.nn.ModuleList(self.experts_list)
        self.add_module("experts_list", self.experts_list)

        # Selecting the method to be used for determining the
        # gating weights
        if self.is_gating_trainable is True:
            if self.binary_selection is True:
                self.get_weights = self._get_weights_binary
            else:
                self.get_weights = self._get_weights_bypass
        else:
            self.get_weights = self._get_weights_not_trainable

        self.total_counts = [torch.tensor(np.zeros(self.n_experts)).to(self.device)]
        self.total_counts_tensor = torch.tensor(np.zeros(self.n_experts)).to(self.device)
        self.b = torch.tensor(np.zeros(self.n_experts)).to(self.device)

        if load_balancing:
            self.adjust_weights = self.load_balancing
        else:
            self.adjust_weights = lambda x, y: x

        if self.use_reshaping:
            self.reshaping = lambda x: x.reshape(x.shape[0], x.shape[1],
                                                 -1).permute(0, 2,
                                                             1).mean(dim=1)
        else:
            self.reshaping = lambda x: x

    def _get_weights_bypass(self, gating: torch.Tensor = None) -> torch.Tensor:
        """When the gating weights are trainable and no post-processing operation
        is applied over them.

        Args:
            gating (torch.Tensor, optional): (Default value = None)

        Returns:
            : The binary weights based on the clusters.
        """

        return gating

    def _get_weights_binary(self, gating: torch.Tensor = None) -> torch.Tensor:
        """Even when the gating weights are trainable, they can be forced to became
        binary.

        Args:
            gating (torch.Tensor, optional): (Default value = None)

        Returns:
            torch.Tensor: The binary weights based on the clusters.
        """

        maxs = torch.max(gating, dim=1).values[:, None]

        return torch.where(gating == maxs, 1.0, 0.0)

    def _get_weights_not_trainable(self, gating: torch.Tensor = None) -> torch.Tensor:
        """When the gating process is not trainable, it is considered some kind of
        clustering approach, which will return integers corresponding to the
        cluster for each sample in the batch

        Args:
            gating (torch.Tensor, optional): (Default value = None)

        Returns:
            torch.Tensor: The binary weights based on the clusters.
        """

        batches_size = gating.shape[0]

        weights = torch.zeros(batches_size, self.n_experts)

        weights[
            np.arange(batches_size).astype(int).tolist(), gating.to(int).tolist()
        ] = 1

        return weights

    def load_balancing(self, weights, counts):

        if self.total_counts_tensor.shape[0] > 1:
            error = torch.mean(self.total_counts_tensor, dim=0) - counts
            self.b = self.b + self.alpha*torch.sign(error)

        weights_ = weights + self.b
        weights_ = self.topk(weights_)
        _weights = torch.where(weights_ != 0, weights, 0)

        return _weights

    def count_assigned_tokens(self, weights):

        weights_ = torch.where(weights > 0, 1, 0)
        count = torch.sum(weights_, dim=0)
        return count 

    def topk(self, weights):

        ktop = torch.topk(weights, self.k, dim=1)
        ktop_values = ktop.values
        ktop_indices = ktop.indices
        weights_ = torch.zeros_like(weights)

        for i in range(len(ktop_indices)):
            weights_[i][ktop_indices[i]] = ktop_values[i]

        return weights_

    def gate(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Gating (routing) the input, it means, attributing a weight for the output of
        each expert, which will be used for the allreduce operation executed on top
        of the MoE model.

        Args:
            input_data (Union[np.ndarray, torch.Tensor]):

        Returns:
            torch.Tensor: The penalties used for weighting the input distributed among the experts.
        """

        gating = self.gating_network.forward(Flatten()(input_data))

        gating_weights_ = self.get_weights(gating=gating)

        gating_weights = self.topk(gating_weights_)

        return gating_weights

    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """Forward method

        Args:
            input_data (Union[np.ndarray, torch.Tensor]): Data to be evaluated using the MoE object.
            **kwargs:

        Returns:
            torch.Tensor: The output of the MoE evaluation.
        """
        input_data = self.reshaping(input_data)
        print(input_data.shape)
        gating_weights_ = self.gate(input_data)
        counts = self.count_assigned_tokens(gating_weights_)
        gating_weights_ = self.adjust_weights(gating_weights_, counts)
        gating_weights = [g[..., None] for g in torch.split(gating_weights_, 1, dim=1)]

        def _forward(worker = None) -> torch.Tensor:
            return worker.forward(input_data, *args, **kwargs)

        output = list(map(_forward, self.experts_list))


        result = sum([g * o for g, o in zip(gating_weights, output)])

        self.total_counts.append(counts)
        self.total_counts_tensor = torch.stack(self.total_counts, dim=0)

        return result 

    def summary(self) -> None:
        """It prints a general view of the architecture."""

        print(self)


