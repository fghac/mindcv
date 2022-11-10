''' Model training pipeline '''
import os
import time
import logging
import mindspore as ms
import numpy as np
import moxing as mox
from mindspore import nn, Tensor, context
from mindspore import FixedLossScaleManager, Model
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import StateMonitor, Allreduce
from config import parse_args

ms.set_seed(1)
np.random.seed(1)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
h1 = logging.StreamHandler()
formatter1 = logging.Formatter('%(message)s',)
logger.addHandler(h1)
h1.setFormatter(formatter1)

### Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))

    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    #Set a cache file to determine whether the data has been copied to obs. 
    #If this file exists during multi-card training, there is no need to copy the dataset multiple times.

    f = open("/cache/download_input.txt", 'w')    
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")
    return

### Copy the output to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return

def DownloadFromQizhi(obs_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ObsToEnv(obs_data_url,data_dir)
        context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)


    if device_num > 1:
        # set device_id and init for multi-card training
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)

        init()
        #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank=int(os.getenv('RANK_ID'))
        if local_rank%8==0:
            ObsToEnv(obs_data_url,data_dir)
        #If the cache file does not exist, it means that the copy data has not been completed,
        #and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)
    return

def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank=int(os.getenv('RANK_ID'))
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank%8==0:
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

    if args.num_classes is None:
        num_classes = dataset_train.num_classes()
    else:
        num_classes = args.num_classes

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
        #  cutmix=args.cutmix,
        #cutmix_prob=args.cutmix_prob,
        num_classes=num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )
    # TODO: fix val_while_train on PYNATIVE_MODE.
    if args.val_while_train and args.mode == 0:
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
            #mixup=args.mixup,
            #cutmix=args.cutmix,
            transform=transform_list_eval,
            num_parallel_workers=args.num_parallel_workers,
        )
        # validation dataset count
        eval_count = dataset_eval.get_dataset_size()
    else:
        loader_eval = None

    num_batches = loader_train.get_dataset_size()
    # Train dataset count
    train_count = dataset_train.get_dataset_size()
    if args.distribute:
        all_reduce = Allreduce()
        train_count = all_reduce(Tensor(train_count, ms.int32))

    # create model
    network = create_model(model_name=args.model,
                           num_classes=num_classes,
                           in_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)
    

    
    num_params = sum([param.size for param in network.get_parameters()])

    # create loss
    loss = create_loss(name=args.loss,
                       reduction=args.reduction,
                       label_smoothing=args.label_smoothing,
                       aux_factor=args.aux_factor)

    # create learning rate schedule
    lr_scheduler = create_scheduler(num_batches,
                                    scheduler=args.scheduler,
                                    lr=args.lr,
                                    min_lr=args.min_lr,
                                    warmup_epochs=args.warmup_epochs,
                                    decay_epochs=args.decay_epochs,
                                    decay_rate=args.decay_rate,
                                    milestones=args.multi_step_decay_milestones,
                                    num_epochs=args.epoch_size)
    
    # resume training if ckpt_path is given
    if args.ckpt_path != '' and args.resume_opt: 
        opt_ckpt_path = os.path.join(args.ckpt_save_dir, f'optim_{args.model}.ckpt')
    else:
        opt_ckpt_path = '' 

    # create optimizer
    #TODO: consistent naming opt, name, dataset_name
    optimizer = create_optimizer(network.trainable_params(),
                                 opt=args.opt,
                                 lr=lr_scheduler,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum,
                                 nesterov=args.use_nesterov,
                                 filter_bias_and_bn=args.filter_bias_and_bn,
                                 loss_scale=args.loss_scale,
                                 checkpoint_path=opt_ckpt_path)

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
    # save checkpoint, summary training loss
    # recorad val acc and do model selection if val dataset is availabe
    begin_epoch = 0
    if args.ckpt_path != '':
        if args.ckpt_path != '':
            begin_step = optimizer.global_step.asnumpy()[0]
            begin_epoch = args.ckpt_path.split('/')[-1].split('-')[1].split('_')[0]
            begin_epoch = int(begin_epoch)

    summary_dir = f"./{args.ckpt_save_dir}/summary"
    assert (args.ckpt_save_policy != 'top_k' or args.val_while_train == True), \
        "ckpt_save_policy is top_k, val_while_train must be True."
    state_cb = StateMonitor(model, summary_dir=summary_dir,
                            dataset_val=loader_eval,
                            val_interval=args.val_interval,
                            metric_name="Top_1_Accuracy",
                            ckpt_dir=args.ckpt_save_dir,
                            ckpt_save_interval=args.ckpt_save_interval,
                            best_ckpt_name=args.model + '_best.ckpt',
                            #dataset_sink_mode=args.dataset_sink_mode,
                            rank_id=rank_id,
                            log_interval=args.log_interval,
                            keep_checkpoint_max=args.keep_checkpoint_max,
                            model_name=args.model,
                            last_epoch=begin_epoch,
                            ckpt_save_policy=args.ckpt_save_policy)

    callbacks = [state_cb]
    # log
    if rank_id in [None, 0]:
        logger.info(f"-" * 40)
        logger.info(f"Num devices: {device_num if device_num is not None else 1} \n"
                    f"Distributed mode: {args.distribute} \n"
                    f"Num training samples: {train_count}")
        if args.val_while_train and args.mode == 0:
            logger.info(f"Num validation samples: {eval_count}")
        logger.info(f"Num classes: {num_classes} \n"
                    f"Num batches: {num_batches} \n"
                    f"Batch size: {args.batch_size} \n"
                    f"Auto augment: {args.auto_augment} \n"
                    f"Model: {args.model} \n"
                    f"Model param: {num_params} \n"
                    f"Num epochs: {args.epoch_size} \n"
                    f"Optimizer: {args.opt} \n"
                    f"LR: {args.lr} \n"
                    f"LR Scheduler: {args.scheduler}")
        logger.info(f"-" * 40)

        if args.ckpt_path != '':
            logger.info(f"Resume training from {args.ckpt_path}, last step: {begin_step}, last epoch: {begin_epoch}")
        else:
            logger.info('Start training')

    model.train(args.epoch_size, loader_train, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)

if __name__ == '__main__':

    args = parse_args()
    data_dir = '/cache/data'  
    train_dir = '/cache/output'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)

    ###Initialize and copy data to training image
    DownloadFromQizhi(args.data_url, data_dir)

    data_url = args.data_url
    local_data_path = '/cache/dataset'
    os.makedirs(local_data_path, exist_ok=True)

    from moxing_adapter import sync_data
#如果第一个网址是远程网址，第二个网址是本地路径，则从远程obs下载数据到本地目录 相比之下，将数据从本地目录上传到远程obs。
    sync_data(data_url, local_data_path, threads=256)
#print（value， ...， sep=' '， end='n'， file=sys.stdout， flush=False）

# 默认情况下，将值打印到流或 sys.stdout。可选关键字参数：
# 文件：类似文件的对象（流）;默认为当前 sys.stdout。
# SEP：在值之间插入字符串，默认为空格。
# end：字符串追加在最后一个值之后，默认换行符。
# 冲洗：是否强制冲洗流。
# 相当于重新

    print(f"local_data_path:{os.listdir(local_data_path)}")

    if "imagenet" in os.listdir(local_data_path):
        local_data_path = os.path.join(local_data_path, "imagenet")

    args.data_dir = local_data_path

    ###################################
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        args.ckpt_save_dir = train_dir + "/"
    if device_num > 1:
        args.ckpt_save_dir = train_dir + "/" + str(get_rank()) + "/"
    
    
    train(args)

    ###Copy the trained output data from the local running environment back to obs,
    ###and download it in the training task corresponding to the Qizhi platform
    #This step is not required if UploadOutput is called
    UploadToQizhi(train_dir, args.train_url)
