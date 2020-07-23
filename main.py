import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from data import train_loader, validate_loader
from train_val import validate, train, adjust_learning_rate
from models import models_manager
from utils.common_utils import save_checkpoint


def parse_args(model_names):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('-l', '--logs', dest='logs', action='store_true',
                        help='Save logs')
    parser.add_argument('--logs-dir', dest='logs_dir',
                        help='The directory used to save the logs',
                        default='logs_dir', type=str)
    args = parser.parse_args()
    return args


def main(args, models_mngr):
    best_prec1 = 0

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model = torch.nn.DataParallel(models_mngr.get_model(args.arch))
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    if args.logs:
        if args.logs_dir == 'logs_dir':
            writer = SummaryWriter(f'log_dir/{args.arch}')
        else:
            writer = SummaryWriter(args.logs_dir)
    else:
        writer = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trn_loader = train_loader(args.workers, args.batch_size, normalize)
    val_loader = validate_loader(args.workers, args.batch_size, normalize)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, args.cpu, args.half, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(trn_loader, model, criterion, optimizer, epoch, args.cpu,
              args.half, args.print_freq, writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, args.cpu, args.half,
                         args.print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


if __name__ == '__main__':
    models_mngr = models_manager.Models()
    args = parse_args(models_mngr.get_names())
    main(args, models_mngr)
