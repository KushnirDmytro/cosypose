import random

import numpy as np

from .plotter import Plotter

from cosypose.datasets.wrappers.augmentation_wrapper import AugmentationWrapper
from cosypose.datasets.augmentations import CropResizeToAspectAugmentation
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer

import seaborn as sns

def filter_predictions_by_scene_and_view(preds, scene_id, view_id=None, th=None):
    mask = preds.infos['scene_id'] == scene_id
    if view_id is not None:
        mask = np.logical_and(mask, preds.infos['view_id'] == view_id)
    if th is not None:
        mask = np.logical_and(mask, preds.infos['score'] >= th)
    keep_ids = np.where(mask)[0]
    preds = preds[keep_ids]
    return preds


def make_colormaps(labels):
    colors_hex = sns.color_palette(n_colors=len(labels)).as_hex()
    colormap_hex = {label: color for label, color in zip(labels, colors_hex)}
    colormap_rgb = {k: [int(h[1:][i:i+2], 16) / 255. for i in (0, 2, 4)] + [1.0] for k, h in colormap_hex.items()}
    return colormap_rgb, colormap_hex

def render_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480)):
    pred = pred.cpu()
    camera.update(TWC=np.eye(4))

    colormap_rgb, _ = make_colormaps(pred.infos['label'])
    pred.infos['color'] = pred.infos['label'].apply(lambda k: colormap_rgb[k])

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=row.color,
            TWO=pred.poses[n].numpy(),
        )
        obj['color'][-1] = 0.5   # make transparent
        list_objects.append(obj)

    rendered_scene = renderer.render_scene(list_objects, [camera])
    rgb_rendered = rendered_scene[0]['rgb']
    return rgb_rendered

def render_gt_wrt_camera(renderer, gt, camera=None, resolution=(640, 480)):
    # camera.update(TWC=np.eye(4))
    colormap_rgb, _ = make_colormaps([el['label'] for el in gt])
    for el in gt:
        el['color'] = colormap_rgb[el['label']]
        el['color'][-1] = 0.5  # make transparent

    rendered_scene = renderer.render_scene(gt, [camera])
    rgb_rendered = rendered_scene[0]['rgb']
    return rgb_rendered

def make_singleview_prediction_plots(scene_ds, renderer, predictions, detections=None, resolution=(640, 480), disp_gt=False):
    plotter = Plotter()

    scene_id, view_id = np.unique(predictions.infos['scene_id']).item(), np.unique(predictions.infos['view_id']).item()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])
    idx = scene_ds_index.loc[(scene_id, view_id), 'ds_idx']

    augmentation = CropResizeToAspectAugmentation(resize=resolution)
    scene_ds = AugmentationWrapper(scene_ds, augmentation)
    rgb_input, mask, state = scene_ds[idx]

    figures = dict()

    figures['input_im'] = plotter.plot_image(rgb_input)

    if detections is not None:
        fig_dets = plotter.plot_image(rgb_input)
        fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections)
        figures['detections'] = fig_dets

    pred_rendered = render_prediction_wrt_camera(renderer, predictions, camera=state['camera'])
    figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    figures['pred_overlay'] = plotter.plot_overlay(rgb_input, pred_rendered)

    # TODO: fix GT correct display (need other case when transformation compared)
    if disp_gt:
        gt_rendered = render_gt_wrt_camera(renderer, gt=state['objects'], camera=state['camera'])
        figures['gt_rendered'] = plotter.plot_image(gt_rendered)
        figures['gt_overlay'] = plotter.plot_overlay(rgb_input, gt_rendered)
    return figures
