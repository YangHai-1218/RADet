import json
import argparse
import os 
from os import path as osp
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Convert coco format to bop format')
    parser.add_argument('json_path', type=str)
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = parse_args()
    json_path, save_dir = args.json_path, args.save_dir
    with open(json_path, 'r') as f:
        json_results = json.load(f)
    convert_results = dict()
    for result in json_results:
        scene_id, image_id = result['scene_id'], result['image_id']
        category_id = result['category_id']
        bbox = result['bbox']
        score = result['score']
        if scene_id not in convert_results:
            convert_results[scene_id] = dict()
        if str(image_id) not in convert_results[scene_id]:
            convert_results[scene_id][str(image_id)] = []
        convert_results[scene_id][str(image_id)].append(
            dict(
                bbox_obj=bbox,
                obj_id=category_id,
                score=score,
            )
        )
    
    for scene_id in convert_results:
        save_path = osp.join(save_dir, f"{scene_id:06d}", "scene_gt_info.json")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        mmcv.dump(convert_results[scene_id], save_path)
