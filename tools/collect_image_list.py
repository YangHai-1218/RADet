import argparse
from glob import glob
from os import path as osp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default='data/ycbv/train_real' ,type=str)
    parser.add_argument('--save-path', default='data/ycbv/train_real/train_real.txt', type=str)
    parser.add_argument('--pattern', default='*/rgb/*.png', type=str)
    args = parser.parse_args()
    return args




if __name__ =='__main__':
    args = parse_args()
    image_list = glob(osp.join(args.source_dir, args.pattern))
    image_list = sorted(image_list)
    image_list = [i.replace(args.source_dir+'/', '')+'\n' for i in image_list]
    print(f"Total {len(image_list)} images found")
    with open(args.save_path, 'w') as f:
        f.writelines(image_list)