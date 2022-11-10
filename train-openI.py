''' Model training pipeline '''
import os
import mindspore as ms
import moxing as mox
import json
import time
from mindspore import nn
from mindspore import FixedLossScaleManager, Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication import init, get_rank, get_group_size

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader, sync_data
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import StateMonitor
from config import parse_args

ms.set_seed(1)

code_dir = os.path.dirname(__file__)
work_dir = os.getcwd()
print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))


# Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    # Set a cache file to determine whether the data has been copied to obs.
    # If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    f = open("/cache/download_input.txt", 'w')
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")
    return


# Copy the output to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir, obs_train_url) + str(e))
    return


def DownloadFromQizhi(obs_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ObsToEnv(obs_data_url, data_dir)
        ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    if device_num > 1:
        # set device_id and init for multi-card training
        ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target,
                       device_id=int(os.getenv('ASCEND_DEVICE_ID')))
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel', gradients_mean=True,
                                     parameter_broadcast=True)
        init()
        # Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank = int(os.getenv('RANK_ID'))
        if local_rank % 8 == 0:
            ObsToEnv(obs_data_url, data_dir)
        # If the cache file does not exist, it means that the copy data has not been completed,
        # and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)
    return


def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            EnvToObs(train_dir, obs_train_url)
    return


def train(args):
    ''' main train function'''
    ms.set_context(mode=args.mode)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = None
        rank_id = None

    # create dataset
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.train_split,
        shuffle=args.shuffle,
        num_samples=args.num_samples,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download)

    # create transforms
    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=True,
        image_resize=args.image_resize,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        mean=args.mean,
        std=args.std,
        re_prob=args.re_prob,
        re_scale=args.re_scale,
        re_ratio=args.re_ratio,
        re_value=args.re_value,
        re_max_attempts=args.re_max_attempts
    )

    # load dataset
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        num_classes=args.num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

    if args.val_while_train:
        dataset_eval = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split=args.val_split,
            num_parallel_workers=args.num_parallel_workers,
            download=args.dataset_download)

        transform_list_eval = create_transforms(
            dataset_name=args.dataset,
            is_training=False,
            image_resize=args.image_resize,
            crop_pct=args.crop_pct,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std
        )

        loader_eval = create_loader(
            dataset=dataset_eval,
            batch_size=args.batch_size,
            drop_remainder=False,
            is_training=False,
            transform=transform_list_eval,
            num_parallel_workers=args.num_parallel_workers,
        )
    else:
        loader_eval = None

    steps_per_epoch = loader_train.get_dataset_size()

    # create model
    network = create_model(model_name=args.model,
                           num_classes=args.num_classes,
                           in_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)

    # create loss
    loss = create_loss(name=args.loss,
                       reduction=args.reduction,
                       label_smoothing=args.label_smoothing,
                       aux_factor=args.aux_factor)

    # create learning rate schedule
    lr_scheduler = create_scheduler(steps_per_epoch,
                                    scheduler=args.scheduler,
                                    lr=args.lr,
                                    min_lr=args.min_lr,
                                    warmup_epochs=args.warmup_epochs,
                                    decay_epochs=args.decay_epochs,
                                    decay_rate=args.decay_rate)

    # create optimizer
    # TODO: consistent naming opt, name, dataset_name
    optimizer = create_optimizer(network.trainable_params(),
                                 opt=args.opt,
                                 lr=lr_scheduler,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum,
                                 nesterov=args.use_nesterov,
                                 filter_bias_and_bn=args.filter_bias_and_bn,
                                 loss_scale=args.loss_scale)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy()}

    # init model
    if args.loss_scale > 1.0:
        loss_scale_manager = FixedLossScaleManager(loss_scale=args.loss_scale, drop_overflow_update=False)
        model = Model(network, loss_fn=loss, optimizer=optimizer, metrics=eval_metrics, amp_level=args.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(network, loss_fn=loss, optimizer=optimizer, metrics=eval_metrics, amp_level=args.amp_level)

    # callback
    loss_cb = LossMonitor(per_print_times=steps_per_epoch)
    time_cb = TimeMonitor(data_size=steps_per_epoch)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=int(steps_per_epoch * args.ckpt_save_interval),
        keep_checkpoint_max=args.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=args.model,
                              directory=args.ckpt_save_dir,
                              config=ckpt_config)

    # summary training loss
    # recorad val acc and do model selection if val dataset is availabe
    summary_dir = f"{args.ckpt_save_dir}/summary"
    state_cb = StateMonitor(model, summary_dir=summary_dir,
                            dataset_val=loader_eval,
                            val_interval=args.val_interval,
                            metric_name="Top_1_Accuracy",
                            ckpt_dir=args.ckpt_save_dir,
                            best_ckpt_name=args.model + '_best.ckpt',
                            dataset_sink_mode=args.dataset_sink_mode,
                            rank_id=rank_id)

    callbacks = [loss_cb, time_cb, state_cb]

    if args.distribute:
        if rank_id == 0:
            callbacks.append(ckpt_cb)
    else:
        callbacks.append(ckpt_cb)

    # train model
    model.train(args.epoch_size, loader_train, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)


if __name__ == '__main__':
    args = parse_args()

    data_dir = '/cache/data'
    train_dir = '/cache/output'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)

    # Initialize and copy data to training image
    # DownloadFromQizhi is much slower than sync_data but sync_data usually abort;
    DownloadFromQizhi(args.data_url, data_dir)
    # data_url = args.data_url
    # local_data_path = '/cache/dataset'
    # os.makedirs(local_data_path, exist_ok=True)
    # sync_data(data_url, local_data_path, threads=256)
    print(f"data_dir:{os.listdir(data_dir)}")
    if "imagenet" in os.listdir(data_dir):
        data_dir = os.path.join(data_dir, "imagenet")
    args.data_dir = data_dir

    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        args.ckpt_save_dir = train_dir + "/"
    if device_num > 1:
        args.ckpt_save_dir = train_dir + "/" + str(get_rank()) + "/"

    train(args)

    UploadToQizhi(args.ckpt_save_dir, args.train_url)
