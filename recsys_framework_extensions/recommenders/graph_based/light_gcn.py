import copy
from typing import Optional

import attrs
import numpy as np
import scipy.sparse as sp
import torch
from Recommenders.GraphBased.LightGCNRecommender import normalized_adjacency_matrix
from recsys_framework_extensions.data.io import DataIO
from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.Incremental_Training_Early_Stopping import (
    Incremental_Training_Early_Stopping,
)
from Utils.PyTorch.DataIterator import BPRIterator
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython
from Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
)
from skopt.space import Integer, Categorical, Real
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersLightGCNRecommender(SearchHyperParametersBaseRecommender):
    epochs: Categorical = attrs.field(
        default=Categorical(
            [1500],  # The original paper suggests 1000
        )
    )
    GNN_layers_K: Integer = attrs.field(
        default=Integer(  # The original paper limits it to 4
            low=1,
            high=7,
            prior="uniform",
            base=10,
        )
    )
    batch_size: Categorical = attrs.field(
        default=Categorical(
            categories=[512, 1024, 2048, 4096],
        ),
    )
    embedding_size: Integer = attrs.field(
        default=Integer(
            low=2,
            high=350,
            prior="uniform",
            base=10,
        )
    )
    learning_rate: Real = attrs.field(
        default=Real(
            low=1e-6,
            high=1e-2,
            prior="log-uniform",
            base=10,
        )
    )
    l2_reg: Real = attrs.field(
        default=Real(
            low=1e-6,
            high=1e-2,
            prior="log-uniform",
            base=10,
        )
    )
    dropout_rate: Real = attrs.field(
        default=Real(
            low=0.0,
            high=0.8,
            prior="uniform",
            base=10,
        )
    )
    sgd_mode: Categorical = attrs.field(
        default=Categorical(
            categories=["sgd", "adagrad", "adam", "rmsprop"],
        )
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersLightGCNRecommenderDEBUG(SearchHyperParametersLightGCNRecommender):
    epochs: Categorical = attrs.field(
        default=Categorical(
            [1500],  # The original paper suggests 1000
        )
    )
    GNN_layers_K: Integer = attrs.field(
        default=Categorical(  # The original paper limits it to 4
            [3]
        )
    )
    batch_size: Categorical = attrs.field(
        default=Categorical(
            categories=[2048],
        ),
    )
    embedding_size: Integer = attrs.field(
        default=Categorical(
            [14],
        )
    )
    sgd_mode: Categorical = attrs.field(
        default=Categorical(
            categories=["adam"],
        )
    )



def create_normalized_adjacency_matrix_from_urm(
    urm: sp.csr_matrix,
    add_self_connection: bool = False,
) -> sp.coo_matrix:
    """
    This method creates an adjacency matrix where users and items are nodes and edges represent interactions or impressions. Particularly, all edges are directed ones: an edge is created from a user to an item when the user interacted with the item. An edge is created from an item to a user when the item has been impressed to the user.

    """

    return sp.coo_matrix(
        normalized_adjacency_matrix(
            URM=urm,
            add_self_connection=add_self_connection,
        )
    )


class ExtendedLightGCNModel(torch.nn.Module):
    def __init__(
        self,
        adjacency_matrix: sp.coo_matrix,
        num_users: int,
        num_items: int,
        num_layers: int,
        embedding_size: int,
        dropout_rate: float,
        device: torch.device,
    ):
        super().__init__()

        logger.debug(
            f"\n\t* {adjacency_matrix=}-{adjacency_matrix.shape=}"
            f"\n\t* {num_users=}-{type(num_users)=}"
            f"\n\t* {num_items=}-{type(num_items)=}"
            f"\n\t* {num_layers=}-{type(num_layers)=}"
            f"\n\t* {embedding_size=}-{type(embedding_size)=}"
            f"\n\t* {dropout_rate=}-{type(dropout_rate)=}"
        )

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_users = num_users
        self.num_items = num_items
        self.Graph = torch.sparse_coo_tensor(
            indices=torch.as_tensor(
                np.vstack(
                    [
                        adjacency_matrix.row,
                        adjacency_matrix.col,
                    ]
                )
            ),
            values=adjacency_matrix.data,
            size=adjacency_matrix.shape,
            device=device,
        ).coalesce()
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=embedding_size,
            device=device,
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=embedding_size,
            device=device,
        )

        torch.nn.init.normal_(
            self.embedding_user.weight,
            std=0.1,
        )
        torch.nn.init.normal_(
            self.embedding_item.weight,
            std=0.1,
        )

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if self.dropout_rate > 0.0:
            if self.training:
                g_droped = self.__dropout(1 - self.dropout_rate)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class ExtendedLightGCNRecommender(
    BaseMatrixFactorizationRecommender,
    Incremental_Training_Early_Stopping,
):
    """LightGCNRecommender

    Consider the adjacency matrix   A = |  0     URM |
                                        | URM.T   0  |

    LightGCN learns user and item embeddings E based on a convolution over the graph, the model in matrix form is
    Given D the diagonal matrix containing the node degree, the embeddings are computed as:
    A_tilde = D^(-1/2)*A*D^(-1/2)
    E^(0) = randomly initialized
    E^(k+1) = A_tilde*E^(k)

    E_final = alpha_0 * E^(0) + alpha_1 * E^(1) ... alpha_k * E^(k)
            = alpha_0 * E^(0) + alpha_1 * A_tilde * E^(0) ... alpha_k * A_tilde^k * E^(0)

    In LightGCN E^(0) is trained and alpha can be optimized and learned, but the paper states
    a good predefined value is 1/(K+1)

    @inproceedings{DBLP:conf/sigir/0001DWLZ020,
      author    = {Xiangnan He and
                   Kuan Deng and
                   Xiang Wang and
                   Yan Li and
                   Yong{-}Dong Zhang and
                   Meng Wang},
      editor    = {Jimmy X. Huang and
                   Yi Chang and
                   Xueqi Cheng and
                   Jaap Kamps and
                   Vanessa Murdock and
                   Ji{-}Rong Wen and
                   Yiqun Liu},
      title     = {LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
      booktitle = {Proceedings of the 43rd International {ACM} {SIGIR} conference on
                   research and development in Information Retrieval, {SIGIR} 2020, Virtual
                   Event, China, July 25-30, 2020},
      pages     = {639--648},
      publisher = {{ACM}},
      year      = {2020},
      url       = {https://doi.org/10.1145/3397271.3401063},
      doi       = {10.1145/3397271.3401063},
      timestamp = {Wed, 03 Aug 2022 15:48:33 +0200},
      biburl    = {https://dblp.org/rec/conf/sigir/0001DWLZ020.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """

    RECOMMENDER_NAME = "ExtendedLightGCNRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        use_cython_sampler: bool = True,
        use_gpu: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=verbose,
        )

        self.adjacency_matrix = create_normalized_adjacency_matrix_from_urm(
            urm=self.URM_train,
            add_self_connection=False,
        )

        self.USER_factors: np.ndarray = np.array([])
        self.ITEM_factors: np.ndarray = np.array([])
        self.model_state: dict[str, np.ndarray] = {}

        self.USER_factors_best: np.ndarray = np.array([])
        self.ITEM_factors_best: np.ndarray = np.array([])
        self.model_state_best: dict[str, np.ndarray] = {}

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model: Optional[ExtendedLightGCNModel] = None

        self.batch_size: Optional[int] = None
        self.embedding_size: Optional[int] = None
        self.num_layers: Optional[int] = None

        self.learning_rate: Optional[float] = None
        self.l2_reg: Optional[float] = None
        self.dropout_rate: Optional[float] = None

        self.sgd_mode: Optional[str] = None

        self._data_iterator_class = (
            BPRIterator_cython if use_cython_sampler else BPRIterator
        )

        if use_gpu:
            logger.debug(f"REQUIRED GPU")
            assert torch.cuda.is_available(), "GPU is requested but not available"
            logger.debug(f"REQUIRED GPU - USING CUDA:0")
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            logger.debug(f"REQUIRED CPU - USING CPU:0")
            self.device = torch.device("cpu:0")

    def fit(
        self,
        *,
        epochs: int,
        GNN_layers_K: int,
        batch_size: int,
        embedding_size: int,
        learning_rate: float,
        l2_reg: float,
        dropout_rate: float,
        sgd_mode: str,
        **earlystopping_kwargs,
    ):
        logger.debug(
            f"\n\t* {epochs=}-{type(epochs)=}"
            f"\n\t* {GNN_layers_K=}-{type(GNN_layers_K)=}"
            f"\n\t* {batch_size=}-{type(batch_size)=}"
            f"\n\t* {embedding_size=}-{type(embedding_size)=}"
            f"\n\t* {learning_rate=}-{type(learning_rate)=}"
            f"\n\t* {l2_reg=}-{type(l2_reg)=}"
            f"\n\t* {dropout_rate=}-{type(dropout_rate)=}"
            f"\n\t* {sgd_mode=}-{type(sgd_mode)=}"
        )

        self.batch_size = int(batch_size)
        self.embedding_size = int(embedding_size)
        self.epochs = int(epochs)
        self.num_layers = int(GNN_layers_K)

        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.l2_reg = float(l2_reg)

        self.sgd_mode = str(sgd_mode)

        self._data_iterator = self._data_iterator_class(
            self.URM_train,
            batch_size=self.batch_size,
            set_n_samples_to_draw=self.URM_train.nnz,
        )

        torch.cuda.empty_cache()

        self.model = ExtendedLightGCNModel(
            adjacency_matrix=self.adjacency_matrix,
            num_users=self.n_users,
            num_items=self.n_items,
            num_layers=self.num_layers,
            embedding_size=self.embedding_size,
            dropout_rate=self.dropout_rate,
            device=self.device,
        )
        self.optimizer = get_optimizer(
            optimizer_label=self.sgd_mode.lower(),
            model=self.model,
            learning_rate=self.learning_rate,
            l2_reg=0.0,
        )

        ############################################################
        ### This is a standard training with early stopping part ###
        ############################################################

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()
        self._train_with_early_stopping(
            epochs,
            algorithm_name=self.RECOMMENDER_NAME,
            **earlystopping_kwargs,
        )
        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()
        self.model_state = self.model_state_best

    def _prepare_model_for_validation(self):
        assert self.model is not None

        with torch.no_grad():
            self.model.eval()
            tensor_user_factors, tensor_item_factors = self.model.computer()

            self.USER_factors = tensor_user_factors.detach().cpu().numpy()
            self.ITEM_factors = tensor_item_factors.detach().cpu().numpy()

            self.model_state = clone_pytorch_model_to_numpy_dict(
                self.model,
            )
            self.model.train()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.model_state_best = copy.deepcopy(self.model_state)

    def _run_epoch(self, num_epoch):
        assert self.model is not None
        assert self.optimizer is not None

        if self.verbose:
            batch_iterator = tqdm(self._data_iterator)
        else:
            batch_iterator = self._data_iterator

        epoch_loss = 0
        for batch in batch_iterator:
            # Clear previously computed gradients
            self.optimizer.zero_grad()

            user_batch, pos_item_batch, neg_item_batch = batch
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            # Compute the loss function of the current batch
            loss, reg_loss = self.model.bpr_loss(
                user_batch, pos_item_batch, neg_item_batch
            )
            loss = loss + reg_loss * self.l2_reg

            # Compute gradients given current loss
            loss.backward()
            epoch_loss += loss.item()

            # Apply gradient using the selected _optimizer
            self.optimizer.step()

        self._print("Loss {:.2E}".format(epoch_loss))

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
            "USER_factors": self.USER_factors,
            "ITEM_factors": self.ITEM_factors,
            "model_state": self.model_state,
        }

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save=data_dict_to_save,
        )

        self._print("Saving complete")


__all__ = [
    "ExtendedLightGCNRecommender",
    "SearchHyperParametersLightGCNRecommender",
    "create_normalized_adjacency_matrix_from_urm",
]
