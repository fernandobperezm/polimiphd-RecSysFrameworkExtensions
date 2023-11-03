import os
import socket
import logging
from typing import Any, Callable

import dask.config
from typing_extensions import ParamSpec

import attrs
import distributed
import psutil
import toml
from dask.distributed import Client, LocalCluster
from distributed import Scheduler, Future, as_completed


logger = logging.getLogger(__name__)


_JobParams = ParamSpec("_JobParams")


@attrs.define(frozen=True, kw_only=True)
class DaskConfig:
    use_processes: bool = attrs.field()
    dashboard_address: str = attrs.field()
    scheduler_port: int = attrs.field()
    num_workers: int = attrs.field()
    threads_per_worker: int = attrs.field()
    memory_limit: int = attrs.field()


class DaskInterface:
    def __init__(
        self,
        client: Client,
        config: DaskConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._job_futures: list[Future] = []
        self._job_futures_info: dict[str, dict[str, Any]] = dict()
        self._is_closed: bool = False

    @property
    def client(self) -> Client:
        if self._is_closed:
            raise ValueError(
                "The client's interface is already closed. You need to create a new DaskInterface instance by calling "
                "configure_dask_cluster() again."
            )
        return self._client

    def submit_job(
        self,
        job_key: str,
        job_priority: int,
        job_info: dict[str, Any],
        method: Callable[_JobParams, None],
        method_kwargs: _JobParams.kwargs,
    ):
        job_future = self.client.submit(
            func=method,
            key=job_key,
            priority=job_priority,
            retries=2,
            pure=False,
            allow_other_workers=False,
            actor=False,
            actors=False,
            resources={
                "WORKER": 1,  # Tag used to tell the worker to be completely available for 1 task only.
            },
            **method_kwargs,
        )
        self._job_futures.append(job_future)
        self._job_futures_info[job_future.key] = job_info

    def scatter_data(
        self,
        data: Any,
    ) -> Future:
        return self.client.scatter(
            data=data,
            broadcast=True,
        )

    def wait_for_jobs(self) -> None:
        logger.debug(
            f"Called wait_for_jobs function and the number of future jobs available is {len(self._job_futures)}"
        )
        print(
            f"Called wait_for_jobs function and the number of future jobs available is {len(self._job_futures)}"
        )
        future: Future
        for future in as_completed(self._job_futures):
            experiment_info = self._job_futures_info[future.key]

            try:
                future.result()
                logger.info(f"Successfully finished this job: {experiment_info}")
            except:
                logger.exception(f"The following job failed: {experiment_info}")

    def close(self):
        num_tasks = self.client.run_on_scheduler(_number_of_tasks_in_scheduler)

        if num_tasks > 0:
            logger.info(
                f"Will not shutdown Dask's scheduler due to {num_tasks} running at the moment."
            )
            return

        logger.info(f"Shutting down dask client. No more pending tasks.")

        self._client.close()
        self._is_closed = True


def _load_logger_config() -> DaskConfig:
    with open(os.path.join(os.getcwd(), "pyproject.toml"), "r") as project_file:
        config = toml.load(f=project_file)
    dask_config = DaskConfig(**config["dask"])
    return dask_config


def _is_scheduler_alive(
    scheduler_address: str,
    scheduler_port: int,
) -> bool:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as a_socket:
        try:
            # Avoids exception on binding if the socket was destroyed just before creating it again.
            # See: https://docs.python.org/3/library/socket.html#example
            a_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            a_socket.bind((scheduler_address, scheduler_port))
            return False
        except OSError:
            return True


def configure_dask_cluster() -> DaskInterface:
    dask_global_config = dask.config.global_config
    # dask_resources = dask.config.get("distributed.worker.resources", {})

    machine_memory = psutil.virtual_memory().total
    # Reserve 1GB (2 ** 30) so the machine does not hang or the process gets killed because it
    #  consumes all the available memory.
    machine_memory -= 2**30
    cpu_count = psutil.cpu_count()

    # Recommended by Dask docs:
    #  https://docs.dask.org/en/latest/dataframe-best-practices.html#repartition-to-reduce-overhead
    partition_memory = 100 * 2**20

    # Used to detect if the scheduler is on and to automatically connect to it.
    scheduler_address = "127.0.0.1"

    # if use_processes:
    #     # 4  # Default value for a 16vCPU machine.
    #     n_workers = num_workers
    #
    #     # 4  # Default value for a 16vCPU machine. Cores:= n_workers * threads_per_worker
    #     threads_per_worker = threads_per_worker
    #
    #     # 0 for no limit.  14 * 2 ** 30  # machine_memory / n_workers
    #     # Each worker will have this memory limit.
    #     memory_limit = memory_limit
    # else:
    #     # Default value in Dask's source code
    #     n_workers = num_workers  # 1
    #
    #     # cpu_count  # Default value in Dask's source code
    #     threads_per_worker = threads_per_worker  # cpu_count
    #
    #     # machine_memory  # Default value in Dask's source code
    #     memory_limit = memory_limit  # machine_memory

    config = _load_logger_config()

    if config.use_processes and config.threads_per_worker != 1:
        raise ValueError(
            f"Dask process workers cannot use more than one thread. Value read: {config.threads_per_worker}"
        )

    if not config.use_processes and config.num_workers != 1:
        raise ValueError(
            f"Dask thread workers cannot use more than one worker. Value read: {config.num_workers}"
        )

    if _is_scheduler_alive(
        scheduler_address=scheduler_address, scheduler_port=config.scheduler_port
    ):
        logger.info(
            f"Connecting to already-created scheduler at {scheduler_address}:{config.scheduler_port}"
        )
        client = Client(address=f"{scheduler_address}:{config.scheduler_port}")
    else:
        logger.info(
            f"Dask Client and Cluster information"
            f"\n* {dask_global_config=}"
            # f"\n* {dask_resources=}"
            f"\n* CPU Count={cpu_count}"
            f"\n* Partition Memory={partition_memory / 2 ** 20} MiB"
            f"\n* Installed memory={(machine_memory + 2 ** 30) / 2 ** 30:.2f} GB"
            f"\n* Whole Cluster Usable Memory={machine_memory / 2 ** 30:.2f} GB"
            # f"\n* Worker Usable Memory={config.memory_limit / 2 ** 30:.2f} GB"
            f"\n* {config=}"
        )

        logger.info(f"Creating new Dask Local Cluster Client")
        client = Client(
            LocalCluster(
                processes=config.use_processes,
                n_workers=config.num_workers,
                threads_per_worker=config.threads_per_worker,
                memory_limit=config.memory_limit,
                dashboard_address=config.dashboard_address,
                scheduler_port=config.scheduler_port,
                resources={
                    "WORKER": 1,  # Tag used to tell the worker to be completely available for 1 task only.
                },
            )
        )

    return DaskInterface(
        client=client,
        config=config,
    )


def _number_of_tasks_in_scheduler(dask_scheduler: Scheduler) -> int:
    num_ready_tasks = 0
    ready_task_state = "memory"
    v: distributed.scheduler.TaskState
    for v in dask_scheduler.tasks.values():
        if v.state == ready_task_state:
            num_ready_tasks += 1

    logger.debug(
        f"len(dask_scheduler.tasks)={len(dask_scheduler.tasks)}"
        f"\n* dask_scheduler.n_tasks={dask_scheduler.n_tasks}"
        f"\n* num_ready_tasks={num_ready_tasks}"
    )
    return len(dask_scheduler.tasks)


def close_dask_client(client: Client) -> None:
    num_tasks = client.run_on_scheduler(_number_of_tasks_in_scheduler)

    if num_tasks > 0:
        logger.info(
            f"Will not shutdown Dask's scheduler due to {num_tasks} running at the moment."
        )
        return

    logger.info(f"Shutting down dask client. No more pending tasks.")
    client.close()
