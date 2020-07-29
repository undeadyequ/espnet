"""E2E-TTS training / decoding functions."""

import copy
import json
import logging
import math
import os
import time

import kaldiio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils3.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.utils.training.iterators import ShufflingEnabler

import matplotlib

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter



def train(args):
    """ Training process based on args
    """

    """
    - Set device-related
        - checking gpu available and number 
    - Set data-related
          got idim, odim 
        - Batch preparation
    - Set model-related
        - Model conf logging
        - load pre-trained
        - Check freeze mode (for transfer learning)
        - Resume
    - Set training related
        - save, Eval, Report interval
        - show training info
        - evaluate (att, tensorboard, loss png, )
        - Save attention and model
        - Early stop
    """
    # get I/O dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())

    # Read and Reverse I/O
    idim = int(valid_json[utts[0]]['output'][0]["shape"][1])
    odim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    logging.info("#input dims: " + str(idim))
    logging.info('#output dims: ' + str(odim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),  # !!!
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(idim, odim, args, TTSInterface)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args)
    assert isinstance(model, TTSInterface)
    logging.info(model)
    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu
    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # freeze modules, if specified
    if args.freeze_mods:
        for mod, param in model.state_dict().items():
            if any(key.startswith(mod) for key in args.freeze_mods):
                param.requires_grad = False

    # Setup an optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, eps=args.eps,
            weight_decay=args.weight_decay) # lr schedule ???
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)


    # Batch prepare
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    train_batchset = make_batchset(train_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out, args.minibatches,
                                   batch_sort_key=args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                                   shortest_first=use_sortagrad,
                                   count=args.batch_count,
                                   batch_bins=args.batch_bins,
                                   batch_frames_in=args.batch_frames_in,
                                   batch_frames_out=args.batch_frames_out,
                                   batch_frames_inout=args.batch_frames_inout,
                                   swap_io=True, iaxis=0, oaxis=0)
    valid_batchset = make_batchset(valid_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out, args.minibatches,
                                   batch_sort_key=args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                                   count=args.batch_count,
                                   batch_bins=args.batch_bins,
                                   batch_frames_in=args.batch_frames_in,
                                   batch_frames_out=args.batch_frames_out,
                                   batch_frames_inout=args.batch_frames_inout,
                                   swap_io=True, iaxis=0, oaxis=0)

    train_dataset = dataset_from_batch(train_batchset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size # ???
                                                   shuffle=args.shuffle,
                                                   num_workers=args.workers,
                                                   collate_fn=args.collate_fn)

    valid_dataset = dataset_from_batch(valid_batchset)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=args.batch_size # ???
                                                   shuffle=args.shuffle,
                                                   num_workers=args.workers,
                                                   collate_fn=args.collate_fn)

    updater = Customer

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Set training related
    # Set training related
    #    - save, Eval, Report interval
    #    - show training info
    #    - evaluate (att, tensorboard, loss png, )
    #    - Save attention and model
    #    - Early stop
    # set intervals

    eval_interval = (args.eval_interval_epochs, 'epoch')
    save_interval = (args.save_interval_epochs, 'epoch')
    report_interval = (args.report_interval_iters, 'iteration')

    trainer.extend(
        CustomEvaluator()
    )

    loss_min = 0
    for e in range(args.epochs):
        model.train(True)
        for i, batch in enumerate(valid_dataloader):
            text = batch["text"].to(device)
            mels = batch["mels"].to(device)
            mags = batch["mags"].to(device)

            optimizer.zero_grad()
            loss = model(text, mels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1,)
            optimizer.step()

        # eval
        if e % args.eval_interval_epochs == 0:
            pass  # evaluate (Got loss, synthesized wav)
            # save eval attention map

        # save model
        if e % args.save_interval_epochs == 0:
            if loss < loss_min:
                pass # save the best model
            pass # save

        # set early stop
        pass




class dataset_from_batch(torch.utils.data.Dataset):
    def __init__(self, batches):
        self.batches = batches
    def __len__(self):
        return len(self.batches)
    def __getitem__(self, idx):
        # ...
        txt, mel, mag = self.batches[idx]
        return {"text": txt, "mel": mel, "mag": mag}