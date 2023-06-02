# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
"""Usage: make_web.run(root, 2D_cell_list, wdith) """
from __future__ import print_function
from ast import arg
from glob import glob
import os
import numpy as np
import shutil
from turtle import hideturtle
from flask_table import Table, Col, create_table
from flask import Markup
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=200, help='')
    parser.add_argument('--data_dir', type=str, default='finetune_vizes/*', help='')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='')
    args = parser.parse_args()
    return args
def run(html_root, cell_list, width=200, hide_text=False, height=None):
    """
    cell_list: 2D array, each element could be: filepath of vid/image, str
    """
    os.makedirs(html_root, exist_ok=True)
    ncol = len(cell_list[0])
    # title
    TableCls = create_table('TableCls')
    for c in range(ncol):
        TableCls = TableCls.add_column('%d' % c, Col('%d' % c))
    items = []
    for r, row in enumerate(cell_list):
        line = {}
        for c in range(ncol):
            line['%d' % c] = html_add_col_text(row[c], html_root, width, 'r%02dc%02d' % (r, c), hide_text, height=height)
        items.append(line)
    table = TableCls(items)
    html_str = table.__html__()
    with open(os.path.join(html_root, 'index.html'), 'w') as fp:
        # add header
        fp.write('<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>\n')
        fp.write(html_str)
        print('write to %s.html' % os.path.join(html_root, 'index'))
def html_add_col_text(src_file, vis_dir, width, pref, hide_text=False, height=None):
    """
    :param col_name: cols to add to the line
    :param line:
    :param file_list: list of file to display in (line, col_name)
    :param vis_dir: copy and cache to vis_dir
    :return:
    """
    if height is None:
        size = 'width="%d"' % width
    else:
        size = 'height="%d"' % height
    img_temp = '<a href="{0}"><img src="{0}" %s> </a> <br/> {0} <br/>' % size
    vid_temp = '<video controls %s><source src="{0}" type="video/mp4"></video> <br/> {0} <br/>' % size
    mesh_temp = '<model-viewer src="{0}" style="width:%d" shadow-intensity="1" camera-controls="" auto-rotate="" ar="" ar-status="not-presenting"></model-viewer>' % width
    if hide_text:
        img_temp = img_temp.split('<br/>')[0]
        vid_temp = vid_temp.split('<br/>')[0]
    str_temp = '{0}'
    col_text = ''
    # print(src_file)
    if isinstance(src_file, str) and os.path.exists(src_file):
        ext = src_file.split('.')[-1]
        if ext in ['mp4']:
            temp = vid_temp
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
        elif ext in ['png', 'gif', 'jpg', 'jpeg']:
            temp = img_temp
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
        else:
            dst_file = os.path.join(vis_dir, '%s_%s' % (pref, os.path.basename(src_file)))
            shutil.copyfile(src_file, dst_file)
        col_text += temp.format(os.path.basename(dst_file))
    else:
        col_text += str_temp.format(src_file)
    return Markup(col_text)


def get_cell_list():
    paths = {}
    folders = glob('finetune_vizes/*/alarmclock_lift/*/*eval*',recursive=True)
    folders += glob('finetune_vizes/*/elephant_pass1/*/*eval*',recursive=True)
    for f in folders:
        pstr = os.path.join(*f.split('/')[:-1])
        paths[pstr] = 0
    
    paths = paths.keys()

    

    all_files = []
    for p in paths:
        eval_files = [u for u in os.listdir(p) if ('eval' in u and 'gif' in u)]
        video_file = [int(v.split('-')[-1].split('.')[0][5:]) for v in eval_files]
        max_index = np.argsort(video_file)[-1]
        eval_file = eval_files[max_index]
        eval_path = os.path.join(p,eval_file)
        grasp_path = os.path.join(p,'grasp_obj.gif')
        pregrasp_path = os.path.join(p,'pregrasp.png')
        lo_path = os.path.join(p,'lift_object.gif')
        files = [grasp_path,pregrasp_path,lo_path,eval_path]
        all_files.append(files)

    

    return all_files

if __name__ == '__main__':
    args = parse_args()
    html_root = args.vis_dir

    # html_root = os.path.join(data_dir, args.exp, 'vis_%s' % args.folder)
    print(html_root)
    os.makedirs(html_root, exist_ok=True)
    cell_list = get_cell_list()
    
    print('glob', args.data_dir, len(cell_list), html_root)
    run(
        html_root=html_root,
        cell_list=cell_list,
        width=args.w,
        )