from __future__ import annotations

import gc
import math
import os
from enum import Enum
from typing import Optional, Any

import imageio
import numpy as np
import pandas as pd
import pygifsicle
import scipy.sparse as sp
import seaborn as sns

from Utils.ResultFolderLoader import ResultFolderLoader

from recsys_framework_extensions.logging import get_logger

import matplotlib

matplotlib.use(
    backend='agg',
    force=True
)
import matplotlib.pyplot as plt

logger = get_logger(__name__)

__VALID_CONTEXTS = ["paper", "talk", "notebook", "poster"]

__FONT_FAMILY = "serif"
__FONT_SCALE = 1.0
__PALETTE = "YlGnBu"  # "deep"
__STYLE = "whitegrid"
__CONTEXT = "paper"  # change to "paper" when creating figures for the paper
__FIG_SIZE_WIDTH = (
    16
    if __CONTEXT == "paper"
    else 20
)
__FIG_SIZE_HEIGHT = (
    20
    if __CONTEXT == "paper"
    else 20
)
__FIG_DPI = 300
sns.set_theme(
    context=__CONTEXT,
    style=__STYLE,
    palette=__PALETTE,
    font=__FONT_FAMILY,
    font_scale=__FONT_SCALE,
    rc={
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # 'font.size': 15,
        # 'figure.figsize': (__FIG_SIZE_WIDTH, __FIG_SIZE_HEIGHT),
        'figure.dpi': __FIG_DPI,
        'text.usetex': False,  # True,
        # https://stackoverflow.com/a/64411992/13385583
        'text.latex.preamble': r""" 
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}""",
    }
)
sns.color_palette(
    palette=__PALETTE,
    as_cmap=True
)

assert __CONTEXT in __VALID_CONTEXTS


class FigureSize(Enum):
    ONE_COLUMN_SQUARE = (4, 4)
    ONE_COLUMN_RECTANGLE = (4, 3)
    FULL_PAGE_SQUARE = (8, 8)
    FULL_PAGE_RECTANGLE = (8, 4)
    FULL_PAGE_ONE_THIRD_SQUARE = (8, 2.5)


def _get_training_item_weights_plots_filenames(
    plot_filepath: str,
    norm_plot_filepath: str,
    recommender_name: str,
) -> tuple[str, str, str, str]:
    item_weights_heatmap_with_means_filename = os.path.join(
        plot_filepath,
        f"{recommender_name}_training_item_weights.gif"
    )
    frames_item_weights_heatmap_with_means_base_path = os.path.join(
        plot_filepath,
        "frames",
        f"{recommender_name}_training_item_weights_epoch"
    )

    norm_item_weights_heatmap_with_means_filename = os.path.join(
        norm_plot_filepath,
        f"{recommender_name}_training_item_weights.gif"
    )
    frames_norm_item_weights_heatmap_with_means_base_path = os.path.join(
        norm_plot_filepath,
        "frames",
        f"{recommender_name}_training_item_weights_epoch"
    )

    return (
        item_weights_heatmap_with_means_filename,
        frames_item_weights_heatmap_with_means_base_path,
        norm_item_weights_heatmap_with_means_filename,
        frames_norm_item_weights_heatmap_with_means_base_path,
    )


def _get_item_weights_plots_filenames(
    plot_filepath: str,
    norm_plot_filepath: str,
    recommender_name: str,
) -> tuple[str, str]:
    item_weights_heatmap_with_means_filename = os.path.join(
        plot_filepath,
        f"{recommender_name}_item_weights_with_means.png"
    )
    norm_item_weights_heatmap_with_means_filename = os.path.join(
        norm_plot_filepath,
        f"{recommender_name}_item_weights_with_means.png"
    )

    return item_weights_heatmap_with_means_filename, norm_item_weights_heatmap_with_means_filename


def _get_losses_plots_filenames(
    generator_plots_path: str,
    discriminator_plots_path: str,
    recommender_name: str,
) -> tuple[str, str]:
    generator_losses_filename = os.path.join(
        generator_plots_path,
        f"{recommender_name}_losses_plot.png"
    )

    discriminator_losses_filename = os.path.join(
        discriminator_plots_path,
        f"{recommender_name}_losses_plot.png"
    )

    return generator_losses_filename, discriminator_losses_filename


def _get_item_weights_comparison_plots_filenames(
    norm_plot_filepath: str,
    recommender_names: list[str],
) -> str:
    norm_comparison_all_recommenders_filename = os.path.join(
        norm_plot_filepath,
        f"item_weights_comparison_{'_'.join(recommender_names)}"
    )

    return norm_comparison_all_recommenders_filename


def _get_epochs_comparison_plots_filenames(
    plot_path: str,
    recommender_names: list[str]
) -> tuple[str, str]:
    epochs_comparison_recommenders_filename = os.path.join(
        plot_path,
        f"epochs_comparison_{'_'.join(recommender_names)}"
    )
    epochs_comparison_recommenders_log_scale_filename = os.path.join(
        plot_path,
        f"epochs_comparison_{'_'.join(recommender_names)}_log_scale"
    )

    return epochs_comparison_recommenders_filename, epochs_comparison_recommenders_log_scale_filename


def _get_icm_heatmap_plot_filenames(
    plot_filepath: str,
    icm_name: str,
) -> tuple[str, str]:
    return (
        os.path.join(
            plot_filepath,
            f"{icm_name}_heatmap.png"
        ),
        os.path.join(
            plot_filepath,
            f"norm_{icm_name}_heatmap.png"
        ),
    )


def _get_similarity_heatmap_plot_filenames(
    plot_filepath: str,
    similarity_name: str,
) -> tuple[str, str]:
    return (
        os.path.join(
            plot_filepath,
            f"{similarity_name}_heatmap.png"
        ),
        os.path.join(
            plot_filepath,
            f"norm_{similarity_name}_heatmap.png"
        ),
    )


def _generate_item_weights_with_urm_heatmap(
    urm: sp.csr_matrix,
    user_popularity: np.ndarray,
    item_popularity: np.ndarray,
    item_weights: np.ndarray,
    normalize: bool,
    plot_title: str,
    plot_title_extras: dict[str, str],
    plot_filename: str,
) -> None:
    num_rows = 4
    num_cols = 6
    height_rows_ratios = [5, 75, 10, 10]
    width_cols_ratios = [10, 10, 30, 30, 10, 10]

    fig: plt.Figure = plt.figure(
        figsize=(__FIG_SIZE_WIDTH, __FIG_SIZE_WIDTH),
        dpi=__FIG_DPI
    )
    gs = plt.GridSpec(
        nrows=num_rows,
        ncols=num_cols,
        figure=fig,
        height_ratios=height_rows_ratios,
        width_ratios=width_cols_ratios,
    )

    ax_urm_heatmap_color_bar: plt.Axes = fig.add_subplot(gs[0, 2])
    ax_item_weights_heatmap_color_bar: plt.Axes = fig.add_subplot(gs[0, 3])

    ax_urm_heatmap: plt.Axes = fig.add_subplot(
        gs[1, 2]
    )
    ax_item_weights_heatmap: plt.Axes = fig.add_subplot(
        gs[1, 3],
        sharex=ax_urm_heatmap,
        sharey=ax_urm_heatmap
    )

    ax_urm_user_popularity_boxplot: plt.Axes = fig.add_subplot(
        gs[1, 0],
    )
    ax_urm_user_popularity_scatter: plt.Axes = fig.add_subplot(
        gs[1, 1],
        sharey=ax_urm_heatmap
    )

    ax_urm_item_popularity_scatter: plt.Axes = fig.add_subplot(
        gs[2, 2],
        sharex=ax_urm_heatmap
    )
    ax_urm_item_popularity_boxplot: plt.Axes = fig.add_subplot(
        gs[3, 2],
    )

    ax_item_weights_user_means_scatter: plt.Axes = fig.add_subplot(
        gs[1, 4],
        sharey=ax_item_weights_heatmap
    )
    ax_item_weights_user_means_boxplot: plt.Axes = fig.add_subplot(
        gs[1, 5],
    )

    ax_item_weights_item_means_scatter: plt.Axes = fig.add_subplot(
        gs[2, 3],
        sharex=ax_item_weights_heatmap
    )
    ax_item_weights_item_means_boxplot: plt.Axes = fig.add_subplot(
        gs[3, 3],
    )

    sort_urm_and_item_weights_by_popularity = True
    if sort_urm_and_item_weights_by_popularity:
        popular_user_indices_desc = np.flip(np.argsort(user_popularity))
        popular_item_indices_desc = np.flip(np.argsort(item_popularity))

        urm = urm[popular_user_indices_desc, :][:, popular_item_indices_desc]
        item_weights = item_weights[popular_user_indices_desc, :][:, popular_item_indices_desc]
        user_popularity = user_popularity[popular_user_indices_desc]
        item_popularity = item_popularity[popular_item_indices_desc]

    item_weights_heatmap_data: np.ndarray
    heatmap_data_min: Optional[float]
    heatmap_data_max: Optional[float]

    heatmap_data_min = np.min(item_weights)
    heatmap_data_max = np.max(item_weights)

    if normalize:
        if np.isclose(heatmap_data_min, heatmap_data_max):
            item_weights_heatmap_data = item_weights - heatmap_data_min
        else:
            item_weights_heatmap_data = (item_weights - heatmap_data_min) / (
                heatmap_data_max - heatmap_data_min)

        heatmap_data_min = 0.0
        heatmap_data_max = 1.0
    else:
        item_weights_heatmap_data = item_weights
        heatmap_data_min = None
        heatmap_data_max = None

    user_weights_mean = np.mean(a=item_weights_heatmap_data, axis=1, dtype=np.float64)
    item_weights_mean = np.mean(a=item_weights_heatmap_data, axis=0, dtype=np.float64)

    plot_objects = [
        [
            ax_urm_heatmap_color_bar, ax_urm_heatmap, urm.toarray(), None, None,
            "User-Rating Matrix",
            ax_urm_user_popularity_boxplot, user_popularity, "User Popularity",
            ax_urm_user_popularity_scatter, user_popularity, "User Popularity",
            ax_urm_item_popularity_boxplot, item_popularity, "Item Popularity",
            ax_urm_item_popularity_scatter, item_popularity, "Item Popularity",
        ],
        [
            ax_item_weights_heatmap_color_bar, ax_item_weights_heatmap,
            item_weights_heatmap_data, heatmap_data_min, heatmap_data_max, "User-Item Weights",

            ax_item_weights_user_means_boxplot, user_weights_mean, "User Weights Mean",
            ax_item_weights_user_means_scatter, user_weights_mean, "User Weights Mean",
            ax_item_weights_item_means_boxplot, item_weights_mean, "Item Weights Mean",
            ax_item_weights_item_means_scatter, item_weights_mean, "Item Weights Mean",
        ]
    ]

    num_users, num_items = urm.shape
    for objects in plot_objects:
        (
            ax_heatmap_color_bar, ax_heatmap, heatmap_data, heatmap_min, heatmap_max, heatmap_title,
            ax_user_boxplot, user_boxplot_data, user_boxplot_title,
            ax_user_scatter, user_scatter_data, user_scatter_title,
            ax_item_boxplot, item_boxplot_data, item_boxplot_title,
            ax_item_scatter, item_scatter_data, item_scatter_title,
        ) = objects

        sns.heatmap(
            data=heatmap_data,
            ax=ax_heatmap,
            cmap="YlGnBu",
            cbar_ax=ax_heatmap_color_bar,
            cbar_kws={"orientation": "horizontal"},
            vmin=heatmap_min,
            vmax=heatmap_max,
        )

        sns.boxplot(
            x=user_boxplot_data,
            color="orange",
            ax=ax_user_boxplot,
        )
        sns.scatterplot(
            y=np.arange(num_users),
            x=user_scatter_data,
            color="orange",
            ax=ax_user_scatter,
            linewidth=0
        )

        sns.boxplot(
            y=item_boxplot_data,
            color="red",
            ax=ax_item_boxplot,
        )
        sns.scatterplot(
            x=np.arange(num_items),
            y=item_scatter_data,
            color="red",
            ax=ax_item_scatter,
            linewidth=0
        )

        ax_heatmap.set_xlabel("Item Ids")
        ax_heatmap.set_ylabel("User Ids")

        ax_user_boxplot.tick_params(labelleft=False, labelright=False)
        ax_user_scatter.tick_params(labelleft=False, labelright=False)

        ax_item_boxplot.tick_params(labeltop=False, labelbottom=False)
        ax_item_scatter.tick_params(labeltop=False, labelbottom=False)

        ax_heatmap.set_title(heatmap_title)

        ax_user_boxplot.set_title(user_boxplot_title)
        ax_user_scatter.set_title(user_scatter_title)

        ax_item_boxplot.set_title(item_boxplot_title)
        ax_item_scatter.set_title(item_scatter_title)

    for key, value in plot_title_extras.items():
        plot_title += f"\n* {key}={value}"

    fig.suptitle(
        t=plot_title
    )
    fig.tight_layout()

    plt.savefig(plot_filename)

    fig.clear()
    plt.close(fig=fig)

    gc.collect()


def generate_training_item_weights_plot(
    urm: sp.csr_matrix,
    user_popularity: np.ndarray,
    item_popularity: np.ndarray,
    training_item_weights: dict[str, np.ndarray],
    plot_path: str,
    norm_plot_path: str,
    recommender_name: str,
    plot_title_extras: dict[str, str],
) -> None:
    """
         The plot is expected to be something like this. It is divided in a 3x2 square where
          * The URM heatmap color-bar goes in 0,2
          * The Item-Weights heatmap color-bar goes in 0,3

          * The URM heatmap goes in 1,2
          * The Item-Weights heatmap goes in 1,3

          * The URM User-Popularity Boxplot goes in 1,0
          * The URM User-Popularity Scatter plot goes in 1,1
          * The URM Item-Popularity Boxplot goes in 2,2
          * The URM Item-Popularity Scatter plot goes in 3,2

          * The Item-Weights User-scores Boxplot goes in 1,5
          * The Item-Weights User-scores Scatter plot goes in 1,4
          * The Item-Weights Item-scores Boxplot goes in 2,3
          * The Item-Weights Item-scores Scatter plot goes in 3,3

          * E represent empty cells of the map.

               0           1           2            3            4           5
           -----------------------------------------------------------------------
         0 |   E      |   E      |  URM       | Item-Weights |   E     |   E      |
           |   E      |   E      |  color-bar | color-bar    |   E     |   E      |
           |__________|__________|____________|______________|_________|__________|
           | User-Pop | User-Pop |  URM       | Item-Weights | User-W  | User-W   |
           | Boxplot  | Scatter  |  Heatmap   | Heatmap      | Scatter | Boxplot  |
           |          |          |            |              |         |          |
         1 |          |          |            |              |         |          |
           |          |          |            |              |         |          |
           |          |          |            |              |         |          |
           |__________|__________|____________|______________|_________|__________|
         2 |   E      |   E      |  Item-Pop  |  Item-Weight |   E     |   E      |
           |   E      |   E      |   Scatter  |  Scatter     |   E     |   E      |
           |__________|__________|____________|______________|_________|__________|
         3 |   E      |   E      |  Item-Pop  |  Item-Weight |   E     |   E      |
           |   E      |   E      |   Boxplot  |  Boxplot     |   E     |   E      |
           |__________|__________|____________|______________|_________|__________|
        """
    (
        training_item_weights_heatmap_with_means_filename,
        frames_training_item_weights_heatmap_with_means_base_path,
        norm_training_item_weights_heatmap_with_means_filename,
        frames_norm_training_item_weights_heatmap_with_means_base_path,
    ) = _get_training_item_weights_plots_filenames(
        plot_filepath=plot_path,
        norm_plot_filepath=norm_plot_path,
        recommender_name=recommender_name,
    )

    normalizes = [True, False]
    gif_filenames = [
        norm_training_item_weights_heatmap_with_means_filename,
        training_item_weights_heatmap_with_means_filename
    ]
    plot_file_base_paths = [
        frames_norm_training_item_weights_heatmap_with_means_base_path,
        frames_training_item_weights_heatmap_with_means_base_path
    ]

    for normalize, gif_filename, plot_base_path in zip(
        normalizes,
        gif_filenames,
        plot_file_base_paths
    ):
        if os.path.exists(gif_filename):
            logger.warning(
                f"Skipping plot {gif_filename} because it already exists."
            )
            continue

        epochs = sorted(
            map(
                lambda x: int(x),
                training_item_weights.keys()
            )
        )
        filenames = []

        for epoch in epochs:
            logger.info(f"Running epoch {epoch}")

            plot_title = (
                "Generated Normal User vs Item Weights"
                if normalize
                else "Generated User vs Item Weights"
            )

            plot_title_extras = {
                **plot_title_extras,
                "Current Epoch": str(epoch),
            }

            plot_filename = f"{plot_base_path}_{epoch}.png"
            filenames.append(plot_filename)

            if os.path.exists(plot_filename):
                logger.warning(
                    f"Skipping plot {plot_filename} because it already exists."
                )
                continue

            _generate_item_weights_with_urm_heatmap(
                urm=urm,
                user_popularity=user_popularity,
                item_popularity=item_popularity,
                item_weights=training_item_weights[str(epoch)],
                normalize=normalize,
                plot_title=plot_title,
                plot_title_extras=plot_title_extras,
                plot_filename=plot_filename,
            )

        with imageio.get_writer(gif_filename, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        pygifsicle.optimize(gif_filename)

        logger.info(f"Successfully saved {gif_filename}")

    logger.info(
        f"Created normalized and regular training item weights plot for recommender"
        f" {recommender_name}. "
        f"Plot are located at {training_item_weights_heatmap_with_means_filename} and "
        f"{norm_training_item_weights_heatmap_with_means_filename}"
    )


def generate_item_weights_plot(
    urm: sp.csr_matrix,
    user_popularity: np.ndarray,
    item_popularity: np.ndarray,
    item_weights: np.ndarray,
    plot_path: str,
    norm_plot_path: str,
    recommender_name: str,
    plot_title_extras: dict[str, str],
) -> None:
    """
     The plot is expected to be something like this. It is divided in a 3x2 square where
      * The URM heatmap color-bar goes in 0,2
      * The Item-Weights heatmap color-bar goes in 0,3

      * The URM heatmap goes in 1,2
      * The Item-Weights heatmap goes in 1,3

      * The URM User-Popularity Boxplot goes in 1,0
      * The URM User-Popularity Scatter plot goes in 1,1
      * The URM Item-Popularity Boxplot goes in 2,2
      * The URM Item-Popularity Scatter plot goes in 3,2

      * The Item-Weights User-scores Boxplot goes in 1,5
      * The Item-Weights User-scores Scatter plot goes in 1,4
      * The Item-Weights Item-scores Boxplot goes in 2,3
      * The Item-Weights Item-scores Scatter plot goes in 3,3

      * E represent empty cells of the map.

           0           1           2            3            4           5
       -----------------------------------------------------------------------
     0 |   E      |   E      |  URM       | Item-Weights |   E     |   E      |
       |   E      |   E      |  color-bar | color-bar    |   E     |   E      |
       |__________|__________|____________|______________|_________|__________|
       | User-Pop | User-Pop |  URM       | Item-Weights | User-W  | User-W   |
       | Boxplot  | Scatter  |  Heatmap   | Heatmap      | Scatter | Boxplot  |
       |          |          |            |              |         |          |
     1 |          |          |            |              |         |          |
       |          |          |            |              |         |          |
       |          |          |            |              |         |          |
       |__________|__________|____________|______________|_________|__________|
     2 |   E      |   E      |  Item-Pop  |  Item-Weight |   E     |   E      |
       |   E      |   E      |   Scatter  |  Scatter     |   E     |   E      |
       |__________|__________|____________|______________|_________|__________|
     3 |   E      |   E      |  Item-Pop  |  Item-Weight |   E     |   E      |
       |   E      |   E      |   Boxplot  |  Boxplot     |   E     |   E      |
       |__________|__________|____________|______________|_________|__________|
    """
    (
        item_weights_heatmap_with_means_filename,
        norm_item_weights_heatmap_with_means_filename
    ) = _get_item_weights_plots_filenames(
        plot_filepath=plot_path,
        norm_plot_filepath=norm_plot_path,
        recommender_name=recommender_name,
    )

    for normalize in [True, False]:
        if normalize and os.path.exists(norm_item_weights_heatmap_with_means_filename):
            logger.warning(
                f"Skipping plot {norm_item_weights_heatmap_with_means_filename} because "
                f"it already exists."
            )
            continue

        if not normalize and os.path.exists(item_weights_heatmap_with_means_filename):
            logger.warning(
                f"Skipping plot {item_weights_heatmap_with_means_filename} because "
                f"it already exists."
            )
            continue

        num_rows = 4
        num_cols = 6
        height_rows_ratios = [5, 75, 10, 10]
        width_cols_ratios = [10, 10, 30, 30, 10, 10]

        fig: plt.Figure = plt.figure(
            figsize=(__FIG_SIZE_WIDTH, __FIG_SIZE_HEIGHT),
            dpi=__FIG_DPI
        )
        gs = plt.GridSpec(
            nrows=num_rows,
            ncols=num_cols,
            figure=fig,
            height_ratios=height_rows_ratios,
            width_ratios=width_cols_ratios,
        )

        ax_urm_heatmap_color_bar: plt.Axes = fig.add_subplot(gs[0, 2])
        ax_item_weights_heatmap_color_bar: plt.Axes = fig.add_subplot(gs[0, 3])

        ax_urm_heatmap: plt.Axes = fig.add_subplot(
            gs[1, 2]
        )
        ax_item_weights_heatmap: plt.Axes = fig.add_subplot(
            gs[1, 3],
            sharex=ax_urm_heatmap,
            sharey=ax_urm_heatmap
        )

        ax_urm_user_popularity_boxplot: plt.Axes = fig.add_subplot(
            gs[1, 0],
        )
        ax_urm_user_popularity_scatter: plt.Axes = fig.add_subplot(
            gs[1, 1],
            sharey=ax_urm_heatmap
        )

        ax_urm_item_popularity_scatter: plt.Axes = fig.add_subplot(
            gs[2, 2],
            sharex=ax_urm_heatmap
        )
        ax_urm_item_popularity_boxplot: plt.Axes = fig.add_subplot(
            gs[3, 2],
        )

        ax_item_weights_user_means_scatter: plt.Axes = fig.add_subplot(
            gs[1, 4],
            sharey=ax_item_weights_heatmap
        )
        ax_item_weights_user_means_boxplot: plt.Axes = fig.add_subplot(
            gs[1, 5],
        )

        ax_item_weights_item_means_scatter: plt.Axes = fig.add_subplot(
            gs[2, 3],
            sharex=ax_item_weights_heatmap
        )
        ax_item_weights_item_means_boxplot: plt.Axes = fig.add_subplot(
            gs[3, 3],
        )

        item_weights_heatmap_data: np.ndarray
        heatmap_data_min: Optional[float]
        heatmap_data_max: Optional[float]

        heatmap_data_min = np.min(item_weights)
        heatmap_data_max = np.max(item_weights)

        if normalize:
            if np.isclose(heatmap_data_min, heatmap_data_max):
                item_weights_heatmap_data = item_weights - heatmap_data_min
            else:
                item_weights_heatmap_data = (item_weights - heatmap_data_min) / (
                    heatmap_data_max - heatmap_data_min)

            heatmap_data_min = 0.0
            heatmap_data_max = 1.0
        else:
            item_weights_heatmap_data = item_weights
            heatmap_data_min = None
            heatmap_data_max = None

        user_weights_mean = np.mean(a=item_weights_heatmap_data, axis=1, dtype=np.float64)
        item_weights_mean = np.mean(a=item_weights_heatmap_data, axis=0, dtype=np.float64)

        plot_objects = [
            [
                ax_urm_heatmap_color_bar, ax_urm_heatmap, urm.toarray(), None, None, "User-Rating Matrix",
                ax_urm_user_popularity_boxplot, user_popularity, "User Popularity",
                ax_urm_user_popularity_scatter, user_popularity, "User Popularity",
                ax_urm_item_popularity_boxplot, item_popularity, "Item Popularity",
                ax_urm_item_popularity_scatter, item_popularity, "Item Popularity",
            ],
            [
                ax_item_weights_heatmap_color_bar, ax_item_weights_heatmap,
                item_weights_heatmap_data, heatmap_data_min, heatmap_data_max, "User-Item Weights",

                ax_item_weights_user_means_boxplot, user_weights_mean, "User Weights Mean",
                ax_item_weights_user_means_scatter, user_weights_mean, "User Weights Mean",
                ax_item_weights_item_means_boxplot, item_weights_mean, "Item Weights Mean",
                ax_item_weights_item_means_scatter, item_weights_mean, "Item Weights Mean",
            ]
        ]

        num_users, num_items = urm.shape
        for objects in plot_objects:
            (
                ax_heatmap_color_bar, ax_heatmap, heatmap_data, heatmap_min, heatmap_max, heatmap_title,
                ax_user_boxplot, user_boxplot_data, user_boxplot_title,
                ax_user_scatter, user_scatter_data, user_scatter_title,
                ax_item_boxplot, item_boxplot_data, item_boxplot_title,
                ax_item_scatter, item_scatter_data, item_scatter_title,
            ) = objects

            sns.heatmap(
                data=heatmap_data,
                ax=ax_heatmap,
                cmap="YlGnBu",
                cbar_ax=ax_heatmap_color_bar,
                cbar_kws={"orientation": "horizontal"},
                vmin=heatmap_min,
                vmax=heatmap_max,
            )

            sns.boxplot(
                x=user_boxplot_data,
                color="orange",
                ax=ax_user_boxplot,
            )
            sns.scatterplot(
                y=np.arange(num_users),
                x=user_scatter_data,
                color="orange",
                ax=ax_user_scatter,
                linewidth=0
            )

            sns.boxplot(
                y=item_boxplot_data,
                color="red",
                ax=ax_item_boxplot,
            )
            sns.scatterplot(
                x=np.arange(num_items),
                y=item_scatter_data,
                color="red",
                ax=ax_item_scatter,
                linewidth=0
            )

            ax_heatmap.set_xlabel("Item Ids")
            ax_heatmap.set_ylabel("User Ids")

            ax_user_boxplot.tick_params(labelleft=False, labelright=False)
            ax_user_scatter.tick_params(labelleft=False, labelright=False)

            ax_item_boxplot.tick_params(labeltop=False, labelbottom=False)
            ax_item_scatter.tick_params(labeltop=False, labelbottom=False)

            ax_heatmap.set_title(heatmap_title)

            ax_user_boxplot.set_title(user_boxplot_title)
            ax_user_scatter.set_title(user_scatter_title)

            ax_item_boxplot.set_title(item_boxplot_title)
            ax_item_scatter.set_title(item_scatter_title)

        plot_title = (
            "Generated Normal User vs Item Weights"
            if normalize
            else "Generated User vs Item Weights"
        )
        for key, value in plot_title_extras.items():
            plot_title += f"\n* {key}={value}"

        fig.suptitle(
            t=plot_title
        )
        fig.tight_layout()

        plt.savefig(
            norm_item_weights_heatmap_with_means_filename
            if normalize
            else item_weights_heatmap_with_means_filename
        )

        fig.clear()
        plt.close(fig=fig)

        gc.collect()

        logger.info(
            f"Created normalized and regular item weights plot for recommender {recommender_name}. "
            f"Plot are located at {norm_item_weights_heatmap_with_means_filename} and "
            f"{item_weights_heatmap_with_means_filename}"
        )


def generate_item_weights_comparison_plot(
    item_weights: list[np.ndarray],
    norm_plot_path: str,
    recommender_names: list[str],
    figure_size: FigureSize = FigureSize.FULL_PAGE_RECTANGLE,
) -> None:
    """
    The plot is expected to be something like this. Supposing that we have 3 recommenders in our
    'item_weights_list', then it is a 2 times 3 grid plot.
     * The normalized heatmap for all item weights go in the first row, i.e., 0,:
     * Each item weight i goes in row 1,i.

           0           1        2          3
       ____________________________________________
       |  IW[0]  |  IW[1]  | IW[2] | Item-Weights |
     0 |         |         |       |   Colorbar   |
       |_________|_________|_______|______________|

     These plots are always normalized because if not, they cannot be compared.

    Notes
    -----
     This plot is generated only in PNG because plotting dense URMs in PDF or SVG consumes a lot of disk space and
     CPU time. Please see https://stackoverflow.com/a/47873037/13385583 for a brief explanation of this.
    """
    (
        norm_item_weights_heatmap_with_means_filename
    ) = _get_item_weights_comparison_plots_filenames(
        norm_plot_filepath=norm_plot_path,
        recommender_names=recommender_names,
    )

    if os.path.exists(f"{norm_item_weights_heatmap_with_means_filename}.png"):
        logger.warning(
            f"Skipping plot {norm_item_weights_heatmap_with_means_filename} because "
            f"it already exists."
        )
        return

    if len(recommender_names) != len(item_weights):
        raise ValueError(
            f"Cannot generate plot item weights comparison if the number of recommenders is different than the number "
            f"of item weights. Number of recommenders: {len(recommender_names)}, number of item weights: "
            f"{len(item_weights)}"
        )

    if len(recommender_names) <= 0:
        raise ValueError(
            f"Need at least one recommender to generate item weights comparison"
        )

    def normalize_item_weights(item_weights: np.ndarray):
        heatmap_data_min = np.min(item_weights)
        heatmap_data_max = np.max(item_weights)

        if np.isclose(heatmap_data_min, heatmap_data_max):
            item_weights_heatmap_data = item_weights - heatmap_data_min
        else:
            item_weights_heatmap_data = (item_weights - heatmap_data_min) / (
                heatmap_data_max - heatmap_data_min)

        return item_weights_heatmap_data

    num_recommenders = len(recommender_names)

    num_rows = 1
    num_cols = num_recommenders + 1
    height_rows_ratios = [100]
    width_cols_ratios = [
        *[math.floor(95 / num_recommenders) for _ in range(num_recommenders)],
        5
    ]

    fig: plt.Figure = plt.figure(
        figsize=figure_size.value,
    )
    gs = plt.GridSpec(
        nrows=num_rows,
        ncols=num_cols,
        figure=fig,
        height_ratios=height_rows_ratios,
        width_ratios=width_cols_ratios,
    )

    ax_heatmap_color_bar: plt.Axes = fig.add_subplot(
        gs[0, -1]
    )
    ax_first_recommender: plt.Axes = fig.add_subplot(
        gs[0, 0]
    )
    axs_recommenders: list[plt.Axes] = [
        fig.add_subplot(
            gs[0, x],
            sharey=ax_first_recommender,
        )
        for x in range(1, num_recommenders)
    ]
    axs_recommenders = [
        ax_first_recommender,
        *axs_recommenders
    ]

    heatmap_data_min = 0.0
    heatmap_data_max = 1.0

    for recommender_number, recommender_name, recommender_item_weight, recommender_ax in zip(
        range(0, num_cols),
        recommender_names,
        item_weights,
        axs_recommenders,
    ):

        recommender_norm_item_weight = normalize_item_weights(
            item_weights=recommender_item_weight,
        )

        sns.heatmap(
            data=recommender_norm_item_weight,
            ax=recommender_ax,
            cmap="YlGnBu",
            cbar_ax=ax_heatmap_color_bar,
            cbar_kws={"orientation": "vertical"},
            vmin=heatmap_data_min,
            vmax=heatmap_data_max,
            square=True,
        )

        recommender_ax.set_xlabel("Item Ids")
        recommender_ax.set_title(recommender_name)

        # Only include the y-label in the first recommender plot and remove white space occupied by invisible
        # y-labels and y-ticks of other recommender plots.
        if recommender_number == 0:
            recommender_ax.set_ylabel("User Ids")
        else:
            recommender_ax.tick_params(labelleft=False)
            plt.margins(x=0, y=0)

    if __CONTEXT != "paper":
        fig.suptitle(
            t="Normalized relevance score for all users and items by different recommenders"
        )

    fig.tight_layout()

    plt.savefig(
        f"{norm_item_weights_heatmap_with_means_filename}.png"
    )

    fig.clear()
    plt.close(fig=fig)

    gc.collect()

    logger.info(
        f"Created normalized item weights comparison for recommenders {', '.join(recommender_names)}. "
        f"Plot is located at {norm_item_weights_heatmap_with_means_filename}.png"
    )


def generate_epochs_comparison_plot(
    epochs: list[int],
    plot_path: str,
    recommender_names: list[str],
    figure_size: FigureSize = FigureSize.ONE_COLUMN_SQUARE,
) -> None:
    """
     The plot is expected to be a bar-plot where in the x-axis we place the recommender names and the y-axis
     contains the number of training epochs for that recommender.
    """
    (
        plots_epochs_comparison_filename,
        plots_epochs_comparison_log_scale_filename,
    ) = _get_epochs_comparison_plots_filenames(
        plot_path=plot_path,
        recommender_names=recommender_names,
    )

    if len(recommender_names) != len(epochs):
        raise ValueError(
            f"Cannot generate plot item weights comparison if the number of recommenders is different than the number "
            f"of item weights. Number of recommenders: {len(recommender_names)}, number of epochs: "
            f"{len(epochs)}"
        )

    if len(recommender_names) <= 0:
        raise ValueError(
            f"Need at least one recommender to generate epochs comparison"
        )

    for use_log, filename in zip(
        [True, False],
        [plots_epochs_comparison_log_scale_filename, plots_epochs_comparison_filename]
    ):
        if use_log and os.path.exists(plots_epochs_comparison_log_scale_filename + ".pdf"):
            logger.warning(
                f"Skipping plot {plots_epochs_comparison_log_scale_filename} because "
                f"it already exists."
            )
            continue

        if not use_log and os.path.exists(plots_epochs_comparison_filename + ".pdf"):
            logger.warning(
                f"Skipping plot {plots_epochs_comparison_filename} because "
                f"it already exists."
            )
            continue

        fig: plt.Figure = plt.figure(
            figsize=figure_size.value
        )

        ax_bar_plot: plt.Axes = sns.barplot(
            y=epochs,
            x=recommender_names,
            log=use_log,
            orient="v"
        )
        ax_bar_plot.set_xlabel("Recommenders")
        ax_bar_plot.set_ylabel("Training Epochs")

        for p in ax_bar_plot.patches:
            ax_bar_plot.annotate(
                text=f"{int(p.get_height())}",
                xy=(p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 2),
                textcoords='offset points'
            )

        if __CONTEXT != "paper":
            fig.suptitle(
                t="Number of training epochs for CFGAN and G-CFGAN recommenders"
            )

        fig.tight_layout()

        plt.savefig(
            f"{filename}.png",
        )
        plt.savefig(
            f"{filename}.pdf",
        )

        fig.clear()
        plt.close(fig=fig)

        gc.collect()

    logger.info(
        f"Created epochs comparison for recommenders {', '.join(recommender_names)}. "
        f"Plot is located at {plots_epochs_comparison_log_scale_filename} and {plots_epochs_comparison_filename} with "
        f"'png', 'svg', and 'pdf' extensions."
    )


def generate_losses_plot(
    generator_losses: dict[str, np.ndarray],
    discriminator_losses: dict[str, np.ndarray],
    generator_plots_path: str,
    discriminator_plots_path: str,
    recommender_name: str,
    plot_title_extras: dict[str, Any]
) -> None:
    generator_losses_filename, discriminator_losses_filename = _get_losses_plots_filenames(
        generator_plots_path=generator_plots_path,
        discriminator_plots_path=discriminator_plots_path,
        recommender_name=recommender_name,
    )

    generator_plot_title = f"{recommender_name} Generator Losses"
    discriminator_plot_title = f"{recommender_name} Discriminator Losses"
    for key, value in plot_title_extras.items():
        generator_plot_title += f"\n* {key}={value}"
        discriminator_plot_title += f"\n* {key}={value}"

    for key in generator_losses.keys():
        generator_losses[key] = generator_losses[key].flatten()

    for key in discriminator_losses.keys():
        discriminator_losses[key] = discriminator_losses[key].flatten()

    fig: plt.Figure
    ax: plt.Axes

    generator_best_epoch_idx = None
    discriminator_best_epoch_idx = None
    if ("Best Epoch" in plot_title_extras
        and "Generator #Steps" in plot_title_extras
        and "Discriminator #Steps" in plot_title_extras):
        generator_best_epoch_idx = plot_title_extras["Best Epoch"] * plot_title_extras["Generator #Steps"]
        discriminator_best_epoch_idx = plot_title_extras["Best Epoch"] * plot_title_extras["Discriminator #Steps"]

    plot_args = [
        (
            generator_plot_title,
            generator_losses_filename,
            generator_losses,
            generator_best_epoch_idx
        ),
        (
            discriminator_plot_title,
            discriminator_losses_filename,
            discriminator_losses,
            discriminator_best_epoch_idx
        ),
    ]

    for plot_title, plot_filename, losses, best_epoch_idx in plot_args:

        if os.path.exists(plot_filename):
            logger.warning(
                f"Skipping plot {plot_filename} because it already exists."
            )
            continue

        fig = plt.figure(
            figsize=(__FIG_SIZE_WIDTH, __FIG_SIZE_WIDTH),
            dpi=__FIG_DPI
        )
        fig.suptitle(
            t=plot_title
        )

        ax = sns.scatterplot(data=pd.DataFrame(losses))
        ax.set_xlabel("Epochs & Steps")
        ax.set_ylabel("Binary Cross-Entropy Loss")

        if best_epoch_idx is not None:
            plt.axvline(
                best_epoch_idx,
                color="r",
                linestyle='--',
                linewidth=4.0,
                label="Best Epoch"
            )
        ax.legend()

        fig.tight_layout()

        plt.savefig(plot_filename)

        fig.clear()
        plt.close(fig=fig)

        gc.collect()


def generate_similarity_heatmap_plot(
    similarity: sp.csr_matrix,
    similarity_name: str,
    plot_path: str,
) -> None:
    """
         The plot is expected to be something like this. It is divided in a 3x2 square where
          * The URM heatmap color-bar goes in row 0
          * The URM heatmap goes in row 1

          * The URM User-Popularity Boxplot goes in 2,0
          * The URM User-Popularity Scatter plot goes in 2,1

          * The URM Item-Popularity Boxplot goes in 3,0
          * The URM Item-Popularity Scatter plot goes in 3,1

               0        |    1
            ------------|------------
          0 |        URM            |
            |        color-bar      |
            |_______________________|
            |         URM           |
            |         Heatmap       |
            |                       |
          1 |                       |
            |                       |
            |                       |
            |_______________________|
        """
    sim_filename, norm_sim_filename = _get_similarity_heatmap_plot_filenames(
        plot_filepath=plot_path,
        similarity_name=similarity_name,
    )

    for normalize, filename in zip(
        [True, False],
        [norm_sim_filename, sim_filename]
    ):

        if os.path.exists(filename):
            logger.warning(
                f"Skipping plot {filename} because "
                f"it already exists."
            )
            return

        num_rows = 2
        num_cols = 1
        height_rows_ratios = [2, 98]
        width_cols_ratios = [100]

        fig: plt.Figure = plt.figure(
            figsize=(__FIG_SIZE_WIDTH, __FIG_SIZE_HEIGHT),
            dpi=__FIG_DPI
        )
        gs = plt.GridSpec(
            nrows=num_rows,
            ncols=num_cols,
            figure=fig,
            height_ratios=height_rows_ratios,
            width_ratios=width_cols_ratios,
        )

        ax_similarity_heatmap_color_bar: plt.Axes = fig.add_subplot(
            gs[0, :]
        )
        ax_similarity_heatmap: plt.Axes = fig.add_subplot(
            gs[1, :]
        )

        dense_similarity = similarity.toarray()
        sim_min, sim_max = dense_similarity.min(), dense_similarity.max()

        if normalize:
            if np.isclose(sim_min, sim_max):
                heatmap_data = dense_similarity - sim_min
            else:
                heatmap_data = (dense_similarity - sim_min) / (sim_max - sim_min)

            heatmap_data_min = 0.0
            heatmap_data_max = 1.0
        else:
            heatmap_data = dense_similarity
            heatmap_data_min = None
            heatmap_data_max = None

        heatmap_data[np.isclose(heatmap_data, 0.0)] = np.nan
        sns.heatmap(
            data=heatmap_data,
            ax=ax_similarity_heatmap,
            cmap="YlGnBu",
            cbar_ax=ax_similarity_heatmap_color_bar,
            cbar_kws={"orientation": "horizontal"},
            vmin=heatmap_data_min,
            vmax=heatmap_data_max,
        )

        ax_similarity_heatmap.set_xlabel("Item Ids")
        ax_similarity_heatmap.set_ylabel("Item Ids")

        ax_similarity_heatmap.set_title(similarity_name)

        plot_title = (
            f"Heatmap-similarity_name={similarity_name}-normalize={normalize}"
        )
        # for key, value in plot_title_extras.items():
        #    plot_title += f"\n* {key}={value}"

        fig.suptitle(
            t=plot_title
        )
        fig.tight_layout()

        plt.savefig(filename)

        fig.clear()
        plt.close(fig=fig)

        gc.collect()

        logger.info(
            f"Created heatmap for URM similarity_name={similarity_name}-normalize={normalize}. "
            f"Plot is located at filename={filename}."
        )


def generate_icm_heatmap_plot(
    icm: sp.csr_matrix,
    icm_name: str,
    plot_path: str,
) -> None:
    """
     The plot is expected to be something like this. It is divided in a 3x2 square where
      * The URM heatmap color-bar goes in row 0
      * The URM heatmap goes in row 1

      * The URM User-Popularity Boxplot goes in 2,0
      * The URM User-Popularity Scatter plot goes in 2,1

      * The URM Item-Popularity Boxplot goes in 3,0
      * The URM Item-Popularity Scatter plot goes in 3,1

           0        |    1
        ------------|------------
      0 |        URM            |
        |        color-bar      |
        |_______________________|
        |         URM           |
        |         Heatmap       |
        |                       |
      1 |                       |
        |                       |
        |                       |
        |_______________________|
    """
    icm_heatmap_filename, norm_icm_heatmap_filename = _get_icm_heatmap_plot_filenames(
        plot_filepath=plot_path,
        icm_name=icm_name,
    )
    for normalize, filename in zip(
        [True, False],
        [norm_icm_heatmap_filename, icm_heatmap_filename]
    ):
        if os.path.exists(filename):
            logger.warning(
                f"Skipping plot {filename} because "
                f"it already exists."
            )
            return

        num_rows = 2
        num_cols = 1
        height_rows_ratios = [2, 98]
        width_cols_ratios = [100]

        fig: plt.Figure = plt.figure(
            figsize=(__FIG_SIZE_WIDTH, __FIG_SIZE_HEIGHT),
            dpi=__FIG_DPI
        )
        gs = plt.GridSpec(
            nrows=num_rows,
            ncols=num_cols,
            figure=fig,
            height_ratios=height_rows_ratios,
            width_ratios=width_cols_ratios,
        )

        ax_icm_heatmap_color_bar: plt.Axes = fig.add_subplot(
            gs[0, :]
        )
        ax_icm_heatmap: plt.Axes = fig.add_subplot(
            gs[1, :]
        )

        dense_similarity = icm.toarray()
        sim_min, sim_max = dense_similarity.min(), dense_similarity.max()

        if normalize:
            if np.isclose(sim_min, sim_max):
                heatmap_data = dense_similarity - sim_min
            else:
                heatmap_data = (dense_similarity - sim_min) / (sim_max - sim_min)

            heatmap_data_min = 0.0
            heatmap_data_max = 1.0
        else:
            heatmap_data = dense_similarity
            heatmap_data_min = None
            heatmap_data_max = None

        heatmap_data[np.isclose(heatmap_data, 0.0)] = np.nan

        sns.heatmap(
            data=heatmap_data,
            ax=ax_icm_heatmap,
            cmap="YlGnBu",
            cbar_ax=ax_icm_heatmap_color_bar,
            cbar_kws={"orientation": "horizontal"},
            vmin=heatmap_data_min,
            vmax=heatmap_data_max,
        )

        ax_icm_heatmap.set_xlabel("Features")
        ax_icm_heatmap.set_ylabel("Item Ids")
        ax_icm_heatmap.set_title(icm_name)

        plot_title = (
            f"Heatmap-ICM-{icm_name}"
        )
        # for key, value in plot_title_extras.items():
        #    plot_title += f"\n* {key}={value}"

        fig.suptitle(
            t=plot_title
        )
        fig.tight_layout()

        plt.savefig(filename)

        fig.clear()
        plt.close(fig=fig)

        gc.collect()

        logger.info(
            f"Created heatmap for ICM {icm_name}. "
            f"Plot is located at filename={filename}."
        )


def generate_accuracy_and_beyond_metrics_latex(
    experiments_folder_path: str,
    export_experiments_folder_path: str,
    num_test_users: int,
    base_algorithm_list: list[Any],
    knn_similarity_list: list[Any],
    other_algorithm_list: list[Any],
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    icm_names: Optional[list[str]],
) -> None:
    os.makedirs(
        export_experiments_folder_path,
        exist_ok=True
    )

    accuracy_metrics_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "accuracy_metrics_latex_results.tex"
    )
    beyond_accuracy_metrics_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "beyond_accuracy_metrics_latex_results.tex"
    )
    all_metrics_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "all_metrics_latex_results.tex"
    )
    time_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "time_latex_results.tex"
    )

    result_loader = ResultFolderLoader(
        experiments_folder_path,
        base_algorithm_list=base_algorithm_list,
        other_algorithm_list=other_algorithm_list,
        KNN_similarity_list=knn_similarity_list,
        ICM_names_list=icm_names,
        UCM_names_list=None
    )

    result_loader.generate_latex_results(
        accuracy_metrics_latex_results_filename,
        metrics_list=accuracy_metrics_list,
        cutoffs_list=cutoffs_list,
        table_title=None,
        highlight_best=False
    )

    result_loader.generate_latex_results(
        beyond_accuracy_metrics_latex_results_filename,
        metrics_list=beyond_accuracy_metrics_list,
        cutoffs_list=cutoffs_list,
        table_title=None,
        highlight_best=False
    )

    result_loader.generate_latex_results(
        all_metrics_latex_results_filename,
        metrics_list=all_metrics_list,
        cutoffs_list=cutoffs_list,
        table_title=None,
        highlight_best=False
    )

    result_loader.generate_latex_time_statistics(
        time_latex_results_filename,
        n_evaluation_users=num_test_users,
        table_title=None
    )


import attrs
@attrs.define
class DataFrameResults:
    df_results: pd.DataFrame = attrs.field()
    df_times: pd.DataFrame = attrs.field()
    df_hyper_params: pd.DataFrame = attrs.field()


def generate_accuracy_and_beyond_metrics_pandas(
    experiments_folder_path: str,
    export_experiments_folder_path: str,
    num_test_users: int,
    base_algorithm_list: list[Any],
    knn_similarity_list: list[Any],
    other_algorithm_list: Optional[list[Any]],
    accuracy_metrics_list: list[str],
    beyond_accuracy_metrics_list: list[str],
    all_metrics_list: list[str],
    cutoffs_list: list[int],
    icm_names: Optional[list[str]],
) -> DataFrameResults:
    os.makedirs(
        export_experiments_folder_path,
        exist_ok=True
    )

    accuracy_metrics_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "accuracy_metrics_latex_results.tex"
    )
    beyond_accuracy_metrics_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "beyond_accuracy_metrics_latex_results.tex"
    )
    all_metrics_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "all_metrics_latex_results.tex"
    )
    time_latex_results_filename = os.path.join(
        export_experiments_folder_path,
        "time_latex_results.tex"
    )

    result_loader = ResultFolderLoader(
        experiments_folder_path,
        base_algorithm_list=base_algorithm_list,
        other_algorithm_list=other_algorithm_list,
        KNN_similarity_list=knn_similarity_list,
        ICM_names_list=icm_names,
        UCM_names_list=None
    )

    df_results = result_loader.get_results_dataframe(
        metrics_list=accuracy_metrics_list + beyond_accuracy_metrics_list,
        cutoffs_list=cutoffs_list,
    )

    df_times = result_loader.get_time_statistics_dataframe(
        n_decimals=4,
        n_evaluation_users=num_test_users,
    )

    df_hyper_params = result_loader.get_hyperparameters_dataframe()

    return DataFrameResults(
        df_results=df_results,
        df_times=df_times,
        df_hyper_params=df_hyper_params,
    )
