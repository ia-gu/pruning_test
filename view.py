"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56
"""
import argparse
import copy
import h5py
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from src.model import CNNModel
from src.get_dataset import build_dataset
from src.landscape_utils import net_plotter
from src.landscape_utils import projection as proj 
from src.landscape_utils import scheduler
from src.landscape_utils import mpi4pytorch as mpi
from src.landscape_utils import h52vtp
from src.landscape_utils import plot_2D as plot_2d

def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return
    with h5py.File(surf_file, 'w') as f:
        f['dir_file'] = dir_file

        # Create the coordinates(resolutions) at which the function is evaluated
        xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
        f['xcoordinates'] = xcoordinates

        if args.y:
            ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
            f['ycoordinates'] = ycoordinates

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)
    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = eval_loss(net, criterion, dataloader)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()
        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))
    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

def eval_loss(net, criterion, loader):
    correct = 0
    total_loss = 0
    total = 0 # number of samples

    net.cuda()
    net.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

    return total_loss/total, 100.*correct/total


###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1
    torch.cuda.set_device(rank)
    os.makedirs(args.weight_path.replace(args.weight_path.split('/')[-1], 'landscape'), exist_ok=True)
    surf_file = args.weight_path.replace(args.weight_path.split('/')[-1], 'landscape/surface.h5')
    if not os.path.exists(surf_file):
        try:
            args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
            args.ymin, args.ymax, args.ynum = (None, None, None)
            if args.y:
                args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')
        
        model = CNNModel(model=args.model, classes=args.num_classes, pretrained=False)
        model.load_state_dict(torch.load(os.path.join(args.weight_path, args.epoch+'.pth')))
        w = net_plotter.get_weights(model)
        s = copy.deepcopy(model.state_dict())

        dir_file = surf_file.replace('surface.h5', 'directions.h5')
        if rank == 0:
            net_plotter.setup_direction(args, dir_file, model)

        if rank == 0:
            setup_surface_file(args, surf_file, dir_file)

        # wait until master has setup the direction file and surface file
        mpi.barrier(comm)

        # load directions
        d = net_plotter.load_directions(dir_file)
        # calculate the consine similarity of the two directions
        if len(d) == 2 and rank == 0:
            similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
            print('cosine similarity between x-axis and y-axis: %f' % similarity)

        #--------------------------------------------------------------------------
        # Setup dataloader
        #--------------------------------------------------------------------------
        # download CIFAR10 if it does not exit
        train_dataset, _ = build_dataset(args)

        mpi.barrier(comm)
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=2)

        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------
        crunch(surf_file, model, w, s, d, train_loader, 'train_loss', 'train_acc', comm, rank, args)
        
    plot_2d.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
    h52vtp.h5_to_vtp(surf_file, 'train_loss', zmax=10)