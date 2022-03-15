import contextlib
import logging
import threading
import time

import dask.dataframe as dd
import hydra
from omegaconf import DictConfig, OmegaConf

from ludwig.api import LudwigModel
# from ludwig.backend import get_local_backend
from ludwig.callbacks import Callback

logger = logging.getLogger(__name__)


MODE = 'ray'
CRITEO_DATASET_PATH = 's3://datasets.us-west-2.predibase.com/criteo/10GB.parquet'
DATASETS_MAP = {
    '100MB': 's3://datasets.us-west-2.predibase.com/criteo/all.parquet/part.0.parquet',
    '1GB': 's3://datasets.us-west-2.predibase.com/criteo/1GB.parquet',
    '10GB': 's3://datasets.us-west-2.predibase.com/criteo/10GB.parquet',
    '100GB': 's3://datasets.us-west-2.predibase.com/criteo/100GB.parquet',
    '1TB': 's3://datasets.us-west-2.predibase.com/criteo/all.parquet',
}


if MODE == 'local':
    CONFIG_DIR = '/Users/shreyarajpal/Predibase/benchmarks/configs/ludwig'
    experiment_name = 'criteo_local'
    # backend_config = {'type': get_local_backend()}
elif MODE == 'ray':
    CONFIG_DIR = '/home/ubuntu/benchmarks/configs/ludwig'
    experiment_name = 'criteo_ray'
    datasets_dir = 's3://datasets.us-west-2.predibase.com'


class TimerCallback(Callback):
    # def __init__(self, queue):
    def __init__(self):
        self.epoch = 0
        # self.queue = queue
        self.train_duration = 0
        self.preproc_duration = 0

    def on_preprocess_start(self, *args, **kwargs):
        self.preproc_start_t = time.time()
        logging.info("Starting Preprocessing Now")

    def on_preprocess_end(self, *args, **kwargs):
        self.preproc_duration = time.time() - self.preproc_start_t
        logging.info("Finished Preprocessing")
        logging.info(f"Total Preprocessing Time: {self.preproc_duration}")

    def on_train_start(self, *args, **kwargs):
        self.train_start_t = time.time()

    def on_train_end(self, output_directory):
        self.train_duration = time.time() - self.train_start_t
        logging.info(f"Total Preprocessing Time: {self.preproc_duration}")
        logging.info(f"Total Training Time: {self.train_duration}")

    def on_epoch_start(self, trainer, progress_tracker, save_path):
        self.epoch += 1
        self.epoch_start_t = time.time()

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        epoch_duration = time.time() - self.epoch_start_t
        # self.queue.put((self.epoch, epoch_duration))


@contextlib.contextmanager
def timeit(duration=None):
    start_t = time.time()
    try:
        yield
    finally:
        if duration is not None:
            duration.append(round(time.time() - start_t, 2))


@hydra.main(config_path=CONFIG_DIR, config_name="criteo")
def app(cfg: DictConfig) -> None:

    logging.warning(f"Experiment: {experiment_name}")

    # from ray.util.queue import Queue as RayQueue
    # queue = RayQueue(actor_options={"num_cpus": 0})

    time_per_epoch = []

    # def read_queue():
        # while True:
            # epoch_num, epoch_time = queue.get()
            # logging.warning(f"Epoch: {epoch_num}, Time: {epoch_time}")
            # time_per_epoch.append(epoch_time)

    # t = threading.Thread(target=read_queue, daemon=True)
    # t.start()

    # timer_callback = TimerCallback(queue)
    timer_callback = TimerCallback()

    config = OmegaConf.to_container(cfg)

    model = LudwigModel(
        config=config,
        logging_level=logging.WARNING,
        callbacks=[timer_callback],
    )

    dataset_size = config['dataset_size']
    # PQ_FILES = [f's3://datasets.us-west-2.predibase.com/criteo/all.parquet/part.{i}.parquet' for i in range(int(dataset_size[:-2]) * 10)]
    # dataset = dd.read_parquet(PQ_FILES)
    dataset = DATASETS_MAP[dataset_size]
    num_workers = config['backend']['trainer']['num_workers']

    duration = []
    with timeit(duration):
        train_stats, _, _ = model.train(
            dataset=dataset,
            # dataset=PQ_FILES,
            experiment_name=f'{experiment_name}_{num_workers}_workers_{dataset_size}',
            model_name='simple_model',
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_processed_output=True,
            skip_save_processed_input=True,
        )

    # time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)

    # print(f'Average time per epoch: {time_per_epoch}')
    # print(f'Samples per sec: {config["trainer"]["batch_size"] / time_per_epoch}')
    print(f'Wall Clock Time: {duration[0]}')
    # print(f'Total number of epochs: {len(train_stats["training"]["combined"])}')
    print('Done')


if __name__ == "__main__":
    app()
