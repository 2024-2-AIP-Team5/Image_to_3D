import logging
import torch
from os import path as osp
import os
from ipg_kit import *
from parser_setter import *
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    from collections import OrderedDict
    opt = OrderedDict([('name', 'SR'), 
                       ('model_type', 'IPGModel'), 
                       ('scale', 4), 
                       ('num_gpu', 1), 
                       ('manual_seed', 10), 
                       ('datasets', 
                        OrderedDict([('test', 
                                      OrderedDict([('name', 'InferenceDataset'), 
                                                   ('type', 'SingleImageDataset'), 
                                                   ('dataroot_lq', 'IPG/basicsr/inputs'), 
                                                   ('io_backend', 
                                                    OrderedDict([('type', 'disk')])), 
                                                    ('phase', 'test'), 
                                                    ('scale', 4)]))])), 
                       ('network_g', 
                        OrderedDict([('type', 'IPG'), 
                                     ('upscale', 4), 
                                     ('in_chans', 3), 
                                     ('img_size', 64), 
                                     ('window_size', 16), 
                                     ('img_range', 1.0), 
                                     ('depths', [6, 6, 6, 6, 6, 6]), 
                                     ('embed_dim', 180), 
                                     ('num_heads', [6, 6, 6, 6, 6, 6]), 
                                     ('mlp_ratio', 4), 
                                     ('upsampler', 'pixelshuffle'), 
                                     ('resi_connection', '1conv'), 
                                     ('graph_flags', [1, 1, 1, 1, 1, 1]), 
                                     ('stage_spec', [['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']]), 
                                     ('dist_type', 'cossim'), ('top_k', 256), ('head_wise', 0), ('sample_size', 32), ('graph_switch', 1), ('flex_type', 'interdiff_plain'), ('FFNtype', 'basic-dwconv3'), ('conv_scale', 0.01), ('conv_type', 'dwconv3-gelu-conv1-ca'), ('diff_scales', [10, 0, 0, 0, 0, 0]), ('fast_graph', 1)])), 
                       ('path', 
                        OrderedDict([('pretrain_network_g', 'IPG_SRx4.pth'), 
                                     ('strict_load_g', False), 
                                     ('results_root', 'IPG/results/'), 
                                     ('log', 'IPG/results/'), 
                                     ('visualization', 'IPG/results/visualization')])), 
                                     ('val', OrderedDict([('save_img', True), ('suffix', 'IPG_X4'), ('metrics', None)])), ('dist', False), ('rank', 0), ('world_size', 1), ('auto_resume', False), ('is_train', False)])
    torch.backends.cudnn.benchmark = True
    # import ipdb;ipdb.set_trace()
    torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    # make_exp_dirs(opt)
    # log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    # logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        # logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)
    
    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        # logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        # add metric print
        if hasattr(model, 'metric_results') and 'psnr' in model.metric_results.keys() and 'ssim' in model.metric_results.keys():
            result_psnr, result_ssim = model.metric_results['psnr'], model.metric_results['ssim']
            os.system(f"echo -n '{result_psnr:.3f},{result_ssim:.4f},' >> all1_results.txt")
            os.system(f"echo -n '{result_psnr:.3f}/{result_ssim:.4f}|' >> all2_results.txt")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
