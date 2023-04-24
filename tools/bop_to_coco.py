import os
import json
import cv2
import numpy as np
from os import path as osp
from tqdm import tqdm
from argparse import ArgumentParser





class_names_cfg = dict(
    icbin=('coffee_cup', 'juice_carton'),
    tudl= ('dragon', 'frog', 'can'),
    lmo=('ape', 'benchvise', 'bowl', 'cam', 'can', 'cat', 'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron','lamp', 'phone'),
    ycbv= ('master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill',  'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick'),
    hb=tuple([i+1 for i in range(33)]),
    itodd=tuple([i+1 for i in range(28)]),
    tless=tuple([i+1 for i in range(30)]),
)

image_resolution_cfg = dict(
    icbin=(640, 480),
    tudl=(640, 480),
    ycbv=(640, 480),
    lmo=(640, 480),
    hb=(640, 480),
    itodd=(1280, 960),
    tless=(720, 540), # train_primesense (400, 400), train_pbr (720, 540)
)


def parse_args():
    parser = ArgumentParser(description='Extract ground annotations from BOP format to COCO format')
    parser.add_argument('--images-dir', default='data/hb/train_pbr', type=str)
    parser.add_argument('--images-list',default='data/hb/image_lists/train_pbr.txt' ,type=str)
    parser.add_argument('--save-path', default='data/hb/detector_annotations/train_pbr.json', type=str)
    parser.add_argument('--segmentation', action='store_true', help='collect segmentation info or not')
    parser.add_argument('--without-gt', action='store_true')
   
    parser.add_argument('--dataset', choices=['icbin', 'tudl', 'tless', 'lmo', 'itodd', 'hb', 'ycbv'])
    args = parser.parse_args()
    return args


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    from skimage import measure
    from shapely.geometry import Polygon, MultiPolygon
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        # if len(contour) < 3:
        #     continue
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        if isinstance(poly, MultiPolygon):
            poly = max(poly, key=lambda a: a.area)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # contours = np.subtract(contours, 1)
    # for contour in contours:
    #     contour = close_contour(contour)
    #     contour = measure.approximate_polygon(contour, tolerance)
    #     if len(contour) < 3:
    #         continue
    #     contour = np.flip(contour, axis=1)
    #     segmentation = contour.ravel().tolist()
    #     # after padding and subtracting 1 we may get -0.5 points in our segmentation
    #     segmentation = [0 if i < 0 else i for i in segmentation]
    #     polygons.append(segmentation)

    return segmentations

def construct_gt_info(sequence_dir, start_end_anno_id, start_end_img_id):
    sequence_gt_info_path = os.path.join(sequence_dir, 'scene_gt_info.json')
    sequence_gt_path = os.path.join(sequence_dir, 'scene_gt.json')
    with open(sequence_gt_info_path, 'r') as f:
        sequence_gt_info = json.load(f)
    with open(sequence_gt_path, 'r') as f:
        sequence_gt = json.load(f)

    image_id, anno_id = start_end_img_id[0], start_end_anno_id[0]
    annos_info = dict()
    pbar = tqdm(sequence_gt_info.keys())
    pbar.set_description(os.path.basename(sequence_dir))
    for id in pbar:
        image_id += 1

        image_path = os.path.join(sequence_dir, 'rgb', id.zfill(6)+'.jpg')
        if os.path.exists(image_path):
            # relative path
            image_path = os.path.join(sequence_dir.split(data_root)[-1], 'rgb', id.zfill(6)+'.jpg')
        else:
            # check png path
            image_path = image_path.replace('jpg', 'png')
            assert os.path.exists(image_path)
            image_path = os.path.join(sequence_dir.split(data_root)[-1], 'rgb', id.zfill(6)+'.png')

        # filter '/'
        image_path = image_path[1:]

        per_img_info = []
        bbox_info_per_image = sequence_gt_info[id]
        category_info_per_image = sequence_gt[id]
        visib_fract_per_image = [f['visib_fract'] for f in bbox_info_per_image]
        bbox_info_per_image = [b[bbox_key] for b in bbox_info_per_image]
        category_info_per_image = [c['obj_id'] for c in category_info_per_image]
        for obj_id, (bbox_info_per_obj, category_info_per_obj, visib_fract_per_obj) in enumerate(zip(bbox_info_per_image, category_info_per_image, visib_fract_per_image)):
            anno_id += 1
            area = bbox_info_per_obj[2] * bbox_info_per_obj[3]
            if seg_collect:
                mask_path = os.path.join(sequence_dir, 'mask_visib', id.zfill(6)+'_'+str(obj_id).zfill(6)+'.png')
                mask_per_obj = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
                mask_per_obj = (mask_per_obj / 255).astype(np.byte)
                polygons = binary_mask_to_polygon(mask_per_obj)
                polygons = [p for p in polygons if len(p)>0]
                if len(polygons) == 0:
                    continue
                per_obj_info = dict(
                    id=anno_id, 
                    image_id=image_id, 
                    category_id=category_info_per_obj, 
                    visib_fract=visib_fract_per_obj,
                    bbox=bbox_info_per_obj,
                    area=area, 
                    iscrowd=0, 
                    segmentation=polygons)
            else:
                per_obj_info = dict(
                    id=anno_id, 
                    image_id=image_id, 
                    category_id=category_info_per_obj, 
                    visib_fract=visib_fract_per_obj,
                    bbox=bbox_info_per_obj,
                    area=area, 
                    iscrowd=0)
            per_img_info.append(per_obj_info)

        annos_info[image_path] = dict(id=image_id, gts_info=per_img_info)
    assert anno_id == start_end_anno_id[1]
    assert image_id == start_end_img_id[1]
    return annos_info
    # with open(os.path.join(sequence_dir, 'collect_gt_info.pkl'), 'wb') as f:
    #     pickle.dump(annos_info, f)


def scan_imageid_and_annoid(sequence_dirs):
    image_start_end_ids = []
    anno_start_end_ids = []
    start_image_id, start_anno_id = 0, 0
    for sequence_dir in sequence_dirs:
        sequence_gt_info_path = osp.join(sequence_dir, 'scene_gt_info.json')
        with open(sequence_gt_info_path, 'r') as f:
            sequence_gt_info = json.load(f)
        image_num = len(sequence_gt_info)
        anno_num = [len(v) for v in sequence_gt_info.values()]
        anno_num = sum(anno_num)
        end_image_id = start_image_id + image_num
        end_anno_id = start_anno_id + anno_num
        image_start_end_ids.append((start_image_id, end_image_id))
        anno_start_end_ids.append((start_anno_id, end_anno_id))
        start_anno_id = end_anno_id
        start_image_id = end_image_id
    return image_start_end_ids, anno_start_end_ids



def make_coco_anno(txt_path, collect_annos, coco_annos_dict):
    with open(txt_path, 'r') as f:
        paths = f.read()
    paths = list(paths.split())

    images_info = []
    annos_info = []
    for path in paths:
        if path in collect_annos:
            images_info.append(dict(file_name=path, id=collect_annos[path]['id'], width=image_w, height=image_h))
            annos_info.extend(collect_annos[path]['gts_info'])

    coco_annos_dict['images'].extend(images_info)
    coco_annos_dict['annotations'].extend(annos_info)
    return coco_annos_dict

def save_test_annotation(txt_file, save_path, category_info):
    annotation = dict()
    images_info = []
    with open(txt_file, 'r') as f:
        image_paths = f.readlines()
    image_id = 0
    for i in range(len(image_paths)):
        image_path = image_paths[i].strip()
        images_info.append(
            dict(file_name=image_path, id=image_id, width=image_w, heigth=image_h)
        )
        image_id += 1
    annotation['images'] = images_info
    annotation['categories'] = category_info
    with open(save_path, 'w') as f:
        json.dump(annotation, f)









if __name__ == '__main__':
    args = parse_args()
    data_root, txt, seg_collect, thread_num = args.images_dir, args.images_list, args.segmentation, args.thread_num
    dataset = args.dataset
    if args.amodal:
        bbox_key = 'bbox_visib'
    else:
        bbox_key = 'bbox_obj'
    
    class_names = class_names_cfg[dataset]
    image_w, image_h = image_resolution_cfg[dataset]

    category_info = []
    # generate category info
    for category_id, category_name in enumerate(class_names):
        category_info.append(dict(id=category_id+1, name=category_name))
    
    if args.without_gt:
        save_test_annotation(txt, args.save_path, category_info)
    else:
        coco_annotations = dict(images=list(), annotations=list(), categories=category_info)
        # generate anno
        collect_info = dict()
        image_id, anno_id = 0, 0
        sequences = sorted(os.listdir(data_root))
        sequences = [osp.join(data_root, s) for s in sequences]
        sequences = [s for s in sequences if osp.isdir(s)]
        image_start_end_ids, anno_start_end_ids = scan_imageid_and_annoid(sequences)
        pbar = tqdm(zip(sequences, image_start_end_ids, anno_start_end_ids))
        for seq, image_start_end_id, anno_start_end_id in pbar:
            collect_info.update(construct_gt_info(seq, anno_start_end_id, image_start_end_id))

        # convert annotations to coco format
        coco_annotations = make_coco_anno(txt, collect_info, coco_annotations)

        with open(args.save_path, 'w') as f:
            json.dump(coco_annotations, f)