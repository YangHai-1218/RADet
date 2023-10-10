import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('bop_test_json', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('--ext', default='png', type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    bop_test_json, save_path, ext = args.bop_test_json, args.save_path, args.ext
    with open(bop_test_json, 'r') as f:
        bop_test = json.load(f)
    image_paths = []
    for obj in bop_test:
        im_id, scene_id = obj['im_id'], obj['scene_id']
        image_path = f"{int(scene_id):06d}/rgb/{int(im_id):06d}.{ext}"
        if image_path in image_paths:
            continue
        else:
            image_paths.append(image_path)
    print(f"total {len(image_paths)} founded")
    with open(save_path, 'w') as f:
        f.writelines([p+'\n' for p in image_paths])

