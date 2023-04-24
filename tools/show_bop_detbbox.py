import json
import argparse
from radet.core.visualization import imshow_det_bboxes
import os
from os import path as osp
import numpy as np

class_names_cfg = dict(
    icbin=('coffee_cup', 'juice_carton'),
    tudl= ('dragon', 'frog', 'can'),
    lmo=('ape', 'benchvise', 'bowl', 'cam', 'can', 'cat', 'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron','lamp', 'phone'),
    ycbv= ('master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill',  'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick'),
    hb=tuple([str(i+1) for i in range(33)]),
    itodd=tuple([str(i+1) for i in range(28)]),
    tless=tuple([str(i+1) for i in range(30)]),
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('result_json')
    parser.add_argument('save_dir')
    parser.add_argument('--show-score-thr', type=float, default=0.3)
    parser.add_argument('--dataset', choices=['icbin', 'itodd', 'ycbv', 'lmo', 'tless', 'hb', 'tudl'])
    parser.add_argument('--ext', default='jpg')
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
    args = parse_args()
    image_dir, result_json, save_dir, show_score_thr, dataset, ext = args.image_dir, args.result_json, args.save_dir, args.show_score_thr, args.dataset, args.ext
    class_names = class_names_cfg[dataset]
    with open(result_json, 'r') as f:
        detect_result = json.load(f)
    
    formated_results = dict()
    for pred in detect_result:
        scene_id, image_id = pred['scene_id'], pred['image_id']
        bbox, score = pred['bbox'], pred['score']
        category_id = pred['category_id']
        if scene_id not in formated_results:
            formated_results[scene_id] = {}
        if image_id not in formated_results[scene_id]:
            formated_results[scene_id][image_id] = {'bbox':[], 'score':[], 'label':[]}
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        formated_results[scene_id][image_id]['bbox'].append(bbox)
        formated_results[scene_id][image_id]['score'].append(score)
        formated_results[scene_id][image_id]['label'].append(category_id)
    
    for scene_id in formated_results:
        for image_id in formated_results[scene_id]:
            image = osp.join(image_dir, f"{scene_id:06d}", "rgb", f"{image_id:06d}.{ext}")
            save_image = osp.join(save_dir, f"{scene_id:06d}", "rgb", f"{image_id:06d}.{ext}")
            os.makedirs(osp.dirname(save_image), exist_ok=True)
            result = formated_results[scene_id][image_id]
            imshow_det_bboxes(
                image,
                np.concatenate([np.array(result['bbox']).reshape(-1, 4), np.array(result['score']).reshape(-1, 1)], axis=-1),
                np.array(result['label']) -1,
                score_thr=show_score_thr,
                show=False,
                out_file=save_image,
                class_names=class_names,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
            )
