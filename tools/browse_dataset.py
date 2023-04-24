import argparse
import os, random
from pathlib import Path

import mmcv
from mmcv import Config

from radet.core.visualization import imshow_det_bboxes
from radet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', default='configs/mask_bop/r50_lmo_cpuassign.py', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=1,
        help='the interval of show (s)')
    parser.add_argument('--type', default='train', type=str)
    parser.add_argument('--random', default=True, type=bool)
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if hasattr(train_data_cfg, 'pipeline'):
        train_data_cfg['pipeline'] = [
            x for x in train_data_cfg.pipeline if x['type'] not in skip_type
        ]
    else:
        train_data_cfg['dataset']['pipeline'] = [
            x for x in train_data_cfg.dataset.pipeline if x['type'] not in skip_type
        ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(getattr(cfg.data, args.type))

    random_index = list(range(len(dataset)))
    if args.random:
        random.shuffle(random_index)

    progress_bar = mmcv.ProgressBar(len(dataset))
    for index in random_index:
        item = dataset[index]
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
            class_names=dataset.CLASSES,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            bbox_color=(255, 102, 61),
            text_color=(255, 102, 61))
        progress_bar.update()


if __name__ == '__main__':
    main()
