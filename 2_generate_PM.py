import gc
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

import configs
from src.utils import pyutils, datautils, imgutils
from src.models.resnet38d_cls import ResNet38ClassificationModel
import src.utils.custom_transformations as CT


def standard_scaling(cams, ln):
    acts = torch.sum(input=cams[ln]['acts'], dim=0)  # (4, 512, 28, 28)
    grads = torch.sum(input=cams[ln]['grads'], dim=0)  # (4, 512, 28, 28)

    cam = torch.mean(input=grads, dim=(-2, -1), keepdim=True)  # (4, 512, 1, 1)
    cam = torch.sum(input=(cam * acts), dim=-3).unsqueeze(0)  # (1, 4, 28, 28)

    cam = F.interpolate(input=cam, size=(224, 224), mode='bilinear', align_corners=False)  # (1, 4, 224, 224)

    mins = torch.amin(input=cam, dim=(-1, -2), keepdim=True)  # (1, 4, 1, 1)
    maxs = torch.amax(input=cam, dim=(-1, -2), keepdim=True)  # (1, 4, 1, 1)
    cam = (cam - mins) / (maxs - mins + 1e-9)  # (1, 4, 224, 224)
    cam = cam - 1e-6  # (1, 4, 224, 224)
    return cam


trans = [
    lambda x: x,  # Original
    # lambda x: torch.rot90(input=x, k=1, dims=(-1, -2)),  # Rotate 90
    # lambda x: torch.rot90(input=x, k=2, dims=(-1, -2)),  # Rotate 180
    # lambda x: torch.rot90(input=x, k=3, dims=(-1, -2)),  # Rotate 270

    lambda x: torch.flip(input=x, dims=(-1,)),  # Horizontal Flip
    # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-1,)), k=1, dims=(-1, -2)),  # Rotate 90
    # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-1,)), k=2, dims=(-1, -2)),  # Rotate 180
    # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-1,)), k=3, dims=(-1, -2)),  # Rotate 270

    lambda x: torch.flip(input=x, dims=(-2,)),  # Vertical Flip
    # # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-2,)), k=1, dims=(-1, -2)),  # Rotate 90
    # # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-2,)), k=2, dims=(-1, -2)),  # Rotate 180
    # # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-2,)), k=3, dims=(-1, -2)),  # Rotate 270

    lambda x: torch.flip(input=x, dims=(-2, -1)),  # Horizontal and Vertical Flip
    # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-2, -1)), k=1, dims=(-1, -2)),  # Rotate 90
    # # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-2, -1)), k=2, dims=(-1, -2)),  # Rotate 180
    # lambda x: torch.rot90(input=torch.flip(input=x, dims=(-2, -1)), k=3, dims=(-1, -2)),  # Rotate 270
]

rev_trans = [
    lambda x: x,  # Original
    # lambda x: torch.rot90(input=x, k=3, dims=(-1, -2)),  # Rotate 270
    # lambda x: torch.rot90(input=x, k=2, dims=(-1, -2)),  # Rotate 180
    # lambda x: torch.rot90(input=x, k=1, dims=(-1, -2)),  # Rotate 90

    lambda x: torch.flip(input=x, dims=(-1,)),  # Horizontal Flip
    # lambda x: torch.flip(input=torch.rot90(input=x, k=3, dims=(-1, -2)), dims=(-1,)),  # Rotate 270
    # lambda x: torch.flip(input=torch.rot90(input=x, k=2, dims=(-1, -2)), dims=(-1,)),  # Rotate 180
    # lambda x: torch.flip(input=torch.rot90(input=x, k=1, dims=(-1, -2)), dims=(-1,)),  # Rotate 90

    lambda x: torch.flip(input=x, dims=(-2,)),  # Vertical Flip
    # # lambda x: torch.flip(input=torch.rot90(input=x, k=3, dims=(-1, -2)), dims=(-2,)),  # Rotate 270
    # # lambda x: torch.flip(input=torch.rot90(input=x, k=2, dims=(-1, -2)), dims=(-2,)),  # Rotate 180
    # # lambda x: torch.flip(input=torch.rot90(input=x, k=1, dims=(-1, -2)), dims=(-2,)),  # Rotate 90

    lambda x: torch.flip(input=x, dims=(-2, -1)),  # Horizontal and Vertical Flip
    # lambda x: torch.flip(input=torch.rot90(input=x, k=3, dims=(-1, -2)), dims=(-2, -1)),  # Rotate 270
    # # lambda x: torch.flip(input=torch.rot90(input=x, k=2, dims=(-1, -2)), dims=(-2, -1)),  # Rotate 180
    # lambda x: torch.flip(input=torch.rot90(input=x, k=1, dims=(-1, -2)), dims=(-2, -1)),  # Rotate 90
]


def main(args):

    os.makedirs(name=args.bgs_path, exist_ok=True)
    os.makedirs(name=args.cams_path, exist_ok=True)
    os.makedirs(name=args.masks_path, exist_ok=True)

    # Initialize dataloaders
    genpmloader = datautils.get_LUAD_HistoSeg_classification_dataloader(
        images_path=args.train_data_path,
        image_names=os.listdir(path=args.train_data_path),
        trans=CT.Compose([
            CT.ToTensor(),
            CT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        batch_size=1,
        shuffle=False
    )

    model = ResNet38ClassificationModel(mu=args.init_mu, num_classes=args.num_classes)
    model.enable_PDA = False
    state_dict = torch.load(f=args.weights, map_location='cpu')
    model.load_state_dict(state_dict=state_dict['model_state_dict'], strict=True)
    model = model.to(device=args.device)
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()
    target_layers = ['b4_3', 'b4_5', 'b5_2', 'b6', 'bn7']

    gcam = imgutils.Grad_CAM(model=model, target_layers=target_layers, device=args.device, num_classes=args.num_classes)

    for i, samples in tqdm(enumerate(genpmloader)):
        name = samples['name'][0]
        label = samples['label']
        image = samples['img']

        if not os.path.exists(path=os.path.join(args.masks_path, 'bn7', name)):

            backgrounds = imgutils.get_backgrounds(
                names=name,
                images=imgutils.imagenet_unnormalizer(image)[0],
                save_path=args.bgs_path
            )
            backgrounds = np.expand_dims(a=backgrounds, axis=(0, 1))

            image = image.repeat(len(trans), 1, 1, 1)
            for i, fn in enumerate(trans):
                image[i] = fn(image[i])

            image = image.to(device=args.device)
            label = label.to(device=args.device)

            cams = gcam(images=image, labels=label, rev_trans=rev_trans)

            for tl in target_layers:
                os.makedirs(name=os.path.join(args.masks_path, tl), exist_ok=True)
                os.makedirs(name=os.path.join(args.cams_path, 'TE', tl), exist_ok=True)
                os.makedirs(name=os.path.join(args.cams_path, 'NEC', tl), exist_ok=True)
                os.makedirs(name=os.path.join(args.cams_path, 'LYM', tl), exist_ok=True)
                os.makedirs(name=os.path.join(args.cams_path, 'TAS', tl), exist_ok=True)

                temp = standard_scaling(cams, tl)
                temp = temp.numpy()
                cv2.imwrite(filename=os.path.join(args.cams_path, 'TE', tl, name), img=temp[0, 0] * 255)
                cv2.imwrite(filename=os.path.join(args.cams_path, 'NEC', tl, name), img=temp[0, 1] * 255)
                cv2.imwrite(filename=os.path.join(args.cams_path, 'LYM', tl, name), img=temp[0, 2] * 255)
                cv2.imwrite(filename=os.path.join(args.cams_path, 'TAS', tl, name), img=temp[0, 3] * 255)

                temp = np.concatenate((temp, backgrounds), axis=1)
                temp = np.argmax(a=temp, axis=1).squeeze(axis=0)
                temp = temp.astype(np.uint8)
                temp = Image.fromarray(obj=temp, mode='P')
                temp.putpalette(data=args.palette)
                temp.save(fp=os.path.join(args.masks_path, tl, name), format='png')

                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    RED = [205, 51, 51]
    GREEN = [0, 255, 0]
    BLUE = [65, 105, 225]
    ORANGE = [255, 165, 0]
    WHITE = [255, 255, 255]

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default=os.getcwd(), type=str)
    parser.add_argument("--session_name", default="Stage_PM", type=str)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--init_mu", default=1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    args.dataset_path = os.path.join(args.root_path, 'datasets', 'LUAD-HistoSeg')
    args.checkpoints_path = os.path.join(args.root_path, 'checkpoints', args.session_name)
    args.train_data_path = os.path.join(args.dataset_path, 'train')
    args.bgs_path = os.path.join(args.checkpoints_path, 'pms', 'bgs')
    args.cams_path = os.path.join(args.checkpoints_path, 'pms', 'cams')
    args.masks_path = os.path.join(args.checkpoints_path, 'pms', 'masks')
    args.weights = os.path.join(args.root_path, 'checkpoints', 'Stage_1', 'weights', 'Stage_1_E8_checkpoint_trained_on_luad.pth')
    args.palette = RED + GREEN + BLUE + ORANGE + WHITE
    args.luad_configs = configs.get_LUAD_HistoSeg_configs(root_path=os.path.join(args.dataset_path))

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # args.device = torch.device('cpu')
    print(args)

    pyutils.set_seed(seed=args.seed)
    datautils.check_LUAD_HistoSeg_dataset(dataset_configs=args.luad_configs)

    main(args=args)