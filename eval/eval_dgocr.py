import sys
import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
from recognizer import TextRecognizer, crop_image
from easydict import EasyDict as edict
from tqdm import tqdm
import os
import torch
import Levenshtein
import numpy as np
import math
import argparse
import random
from PIL import ImageFont
from shutil import copyfile

from dataset_util import load
from t3_dataset import draw_glyph, draw_glyph2, get_caption_pos

PRINT_DEBUG = False
ONLY_FAlSE = True
# num_samples = 4
num_samples = 1
PLACE_HOLDER = '*'
max_chars = 20
max_lines = 20
font = ImageFont.truetype('./font/Arial_Unicode.ttf', size=60)

def draw_pos(ploygon, prob=1.0):
    img = np.zeros((512, 512, 1))
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    return img/255.

def load_data(input_path):
    content = load(input_path)
    d = []
    count = 0
    for gt in content['data_list']:
        info = {}
        info['img_name'] = gt['img_name']
        info['caption'] = gt['caption']
        if PLACE_HOLDER in info['caption']:
            count += 1
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if annotation['valid'] is False:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                pos.append(annotation['pos'])
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    if count > 0:
        print(f"Found {count} image's caption contain placeholder: {PLACE_HOLDER}, change to ' '...")
    return d

def get_item(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']
    item_dict['caption'] = cur_item['caption']
    item_dict['glyphs'] = []
    item_dict['gly_line'] = []
    item_dict['positions'] = []
    item_dict['texts'] = []
    texts = cur_item.get('texts', [])
    if len(texts) > 0:
        sel_idxs = [i for i in range(len(texts))]
        if len(texts) > max_lines:
            sel_idxs = sel_idxs[:max_lines]
        pos_idxs = [cur_item['pos'][i] for i in sel_idxs]
        item_dict['caption'] = get_caption_pos(item_dict['caption'], pos_idxs, 0.0, PLACE_HOLDER)
        item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
        item_dict['texts'] = [cur_item['texts'][i][:max_chars] for i in sel_idxs]
        # glyphs
        for idx, text in enumerate(item_dict['texts']):
            gly_line = draw_glyph(font, text)
            glyphs = draw_glyph2(font, text, item_dict['polygons'][idx], scale=2)
            item_dict['glyphs'] += [glyphs]
            item_dict['gly_line'] += [gly_line]
        # mask_pos
        for polygon in item_dict['polygons']:
            item_dict['positions'] += [draw_pos(polygon, 1.0)]
    fill_caption = False
    if fill_caption:  # if using embedding_manager, DO NOT fill caption!
        for i in range(len(item_dict['texts'])):
            r_txt = item_dict['texts'][i]
            item_dict['caption'] = item_dict['caption'].replace(PLACE_HOLDER, f'"{r_txt}"', 1)
    # padding
    n_lines = min(len(texts), max_lines)
    item_dict['n_lines'] = n_lines
    n_pad = max_lines - n_lines
    if n_pad > 0:
        item_dict['glyphs'] += [np.zeros((512*2, 512*2, 1))] * n_pad
        item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
        item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
    return item_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default='/home/yuxiang.tyx/projects/ControlNet/controlnet_wukong_generated',
        help='path of generated images for eval'
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json',
        help='json path for evaluation dataset'
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default='debug/debug_laion',
        help='json path for evaluation dataset'
    )
    parser.add_argument(
        "--ori_dir",
        type=str,
        default='debug/debug_laion',
        help='json path for evaluation dataset'
    )
    args = parser.parse_args()
    return args


args = parse_args()
img_dir = args.img_dir
input_json = args.input_json
if PRINT_DEBUG:
    vis_dir = args.vis_dir
    ori_dir = args.ori_dir
    os.makedirs(vis_dir, exist_ok=True)

if 'wukong' in input_json:
    model_lang = 'ch'
    rec_char_dict_path = os.path.join('eval/ocr_weights', 'ppocr_keys_v1.txt')
elif 'laion' in input_json:
    rec_char_dict_path = os.path.join('eval/ocr_weights', 'en_dict.txt')


def get_ld(ls1, ls2):
    edit_dist = Levenshtein.distance(ls1, ls2)
    return 1 - edit_dist/(max(len(ls1), len(ls2)) + 1e-5)


def pre_process(img_list, shape):
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    for idx in range(0, img_num):
        # rotate
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        # resize
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode='bilinear',
            align_corners=True,
        )
        # padding
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC ,numpy
    return numpy_list


def main():
    predictor = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
    rec_image_shape = "3, 48, 320"
    args = edict()
    args.rec_image_shape = rec_image_shape
    args.rec_char_dict_path = rec_char_dict_path
    args.rec_batch_num = 1
    args.use_fp16 = False
    text_recognizer = TextRecognizer(args, None)

    data_list = load_data(input_json)
    sen_acc = []
    edit_dist = []
    for i in tqdm(range(len(data_list)), desc='evaluate'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0]
        n_lines = item_dict['n_lines']
        for j in range(num_samples):
            img_path = os.path.join(img_dir, img_name+f'_{j}.jpg')
            # img_path = os.path.join(img_dir, img_name+f'.jpg')
            img = cv2.imread(img_path)
            if PRINT_DEBUG and not ONLY_FAlSE:
                cv2.imwrite(f'{vis_dir}/{i}_{j}.jpg', img)
            if img.shape[0] != 512 or img.shape[1] != 512:
                img = cv2.resize(img, (512, 512))
            ori_img = img
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).float()  # HWC-->CHW
            gt_texts = []
            pred_texts = []
            for k in range(n_lines):  # line
                gt_texts += [item_dict['texts'][k]]
                np_pos = (item_dict['positions'][k]*255.).astype(np.uint8)  # 0-1, hwc
                pred_text = crop_image(img, np_pos)
                pred_texts += [pred_text]
            if n_lines > 0:
                pred_texts = pre_process(pred_texts, rec_image_shape)
                preds_all = []
                for idx, pt in enumerate(pred_texts):
                    if PRINT_DEBUG and not ONLY_FAlSE:
                        cv2.imwrite(f'{vis_dir}/{i}_{j}_{idx}.jpg', pt)
                    rst = predictor(pt)
                    preds_all += [rst['text'][0]]
                for k in range(len(preds_all)):
                    pred_text = preds_all[k]
                    gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in gt_texts[k]]
                    pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in pred_text]
                    if pred_text == gt_texts[k]:
                        sen_acc += [1]
                    else:
                        sen_acc += [0]
                        if PRINT_DEBUG and ONLY_FAlSE:
                            cv2.imwrite(f'{vis_dir}/{i}_{j}_{k}_{gt_texts[k]}.jpg', pred_texts[k])
                            if not os.path.exists(f'{vis_dir}/{i}_{j}.jpg'):
                                cv2.imwrite(f'{vis_dir}/{i}_{j}.jpg', ori_img)
                            img_name = item_dict['img_name']
                            
                            outfile = f'{vis_dir}/{i}_{j}_ori.jpg'
                            if not os.path.exists(outfile):
                                infile = os.path.join(ori_dir, img_name)
                                copyfile(infile, outfile)
                    edit_dist += [get_ld(pred_order, gt_order)]
                    if PRINT_DEBUG:
                        print(f'pred/gt="{pred_text}"/"{gt_texts[k]}", ed={edit_dist[-1]:.4f}')
    print(f'Done, lines={len(sen_acc)}, sen_acc={np.array(sen_acc).mean():.4f}, edit_dist={np.array(edit_dist).mean():.4f}')


if __name__ == "__main__":
    main()
