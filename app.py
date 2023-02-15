#!/usr/bin/env python

from __future__ import annotations

import os
import shlex
import subprocess

import gradio as gr

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'), stdin=f, cwd='ControlNet')
    commands = [
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/dpt_hybrid-midas-501f0c75.pt -O dpt_hybrid-midas-501f0c75.pt',
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/body_pose_model.pth -O body_pose_model.pth',
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/hand_pose_model.pth -O hand_pose_model.pth',
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/mlsd_large_512_fp32.pth -O mlsd_large_512_fp32.pth',
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/mlsd_tiny_512_fp32.pth -O mlsd_tiny_512_fp32.pth',
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/network-bsds500.pth -O network-bsds500.pth',
        'wget https://huggingface.co/ckpt/ControlNet/resolve/main/upernet_global_small.pth -O upernet_global_small.pth',
    ]
    for command in commands:
        subprocess.run(shlex.split(command), cwd='ControlNet/annotator/ckpts/')

from gradio_canny2image import create_demo as create_demo_canny
from gradio_depth2image import create_demo as create_demo_depth
from gradio_fake_scribble2image import create_demo as create_demo_fake_scribble
from gradio_hed2image import create_demo as create_demo_hed
from gradio_hough2image import create_demo as create_demo_hough
from gradio_normal2image import create_demo as create_demo_normal
from gradio_pose2image import create_demo as create_demo_pose
from gradio_scribble2image import create_demo as create_demo_scribble
from gradio_scribble2image_interactive import \
    create_demo as create_demo_scribble_interactive
from gradio_seg2image import create_demo as create_demo_seg
from model import Model

DESCRIPTION = '''# ControlNet

This is an unofficial demo for [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet).
'''
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'''<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.<br/>
<a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>
'''

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Canny'):
            create_demo_canny(model.process_canny)
        with gr.TabItem('Hough'):
            create_demo_hough(model.process_hough)
        with gr.TabItem('HED'):
            create_demo_hed(model.process_hed)
        with gr.TabItem('Scribble'):
            create_demo_scribble(model.process_scribble)
        with gr.TabItem('Scribble Interactive'):
            create_demo_scribble_interactive(
                model.process_scribble_interactive)
        with gr.TabItem('Fake Scribble'):
            create_demo_fake_scribble(model.process_fake_scribble)
        with gr.TabItem('Pose'):
            create_demo_pose(model.process_pose)
        with gr.TabItem('Segmentation'):
            create_demo_seg(model.process_seg)
        with gr.TabItem('Depth'):
            create_demo_depth(model.process_depth)
        with gr.TabItem('Normal map'):
            create_demo_normal(model.process_normal)

demo.queue().launch()
