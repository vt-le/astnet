import os
import natsort
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def make_power_2(n, base=32.0):
    return int(round(n / base) * base)


def get_transform(size, method=Image.BICUBIC, normalize=True, toTensor=True):
    w, h = size
    new_size = [make_power_2(w), make_power_2(h)]

    transform_list = [transforms.Resize(new_size, method)]

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)


class Video(data.Dataset):
    def __init__(self, config):
        super(Video, self).__init__()
        self.new_size = [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
        self.num_frames = config.DATASET.NUM_FRAMES
        frame_steps = config.DATASET.FRAME_STEPS
        frame_steps = min(frame_steps, self.num_frames)
        root = config.DATASET.ROOT
        dataset_name = config.DATASET.DATASET
        train_set = config.DATASET.TRAINSET
        lower_bound = config.DATASET.LOWER_BOUND
        self.dir = os.path.join(root, dataset_name, train_set)
        assert (os.path.exists(self.dir))

        videos = self._colect_filelist(self.dir)

        split_videos = [[video[i:i + self.num_frames]
                         for i in range(0, len(video) // self.num_frames * self.num_frames,
                                        frame_steps if len(video) > lower_bound else 1)]
                        for video in videos]

        self.videos = []
        for video in split_videos:
            for sub_video in video:
                if len(sub_video) == self.num_frames:
                    self.videos.append(sub_video)

        self.num_videos = len(self.videos)

    def _colect_filelist(self, root):
        include_ext = [".png", ".jpg", "jpeg", ".bmp"]
        dirs = [x[0] for x in os.walk(root, followlinks=True)]

        dirs = natsort.natsorted(dirs)

        datasets = [
            [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, el))
             and not el.startswith('.')
             and any([el.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]

        return [el for el in datasets if el]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        video_name = self.videos[index]
        raw_frames = [Image.open(f).convert('RGB') for f in video_name]

        video = []
        for f in raw_frames:
            transform = get_transform(self.new_size)
            f = transform(f)
            video.append(f)

        return {'video': video, 'video_name': video_name}


class TestVideo(data.Dataset):
    def __init__(self, config):
        super(TestVideo, self).__init__()
        self.new_size = [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
        root = config.DATASET.ROOT
        dataset_name = config.DATASET.DATASET
        test_set = config.DATASET.TESTSET
        self.dir = os.path.join(root, dataset_name, test_set)
        assert (os.path.exists(self.dir))

        self.videos = self._colect_filelist(self.dir)

        self.num_videos = len(self.videos)

    def _colect_filelist(self, root):
        include_ext = [".png", ".jpg", "jpeg", ".bmp"]
        dirs = [x[0] for x in os.walk(root, followlinks=True)]

        dirs = natsort.natsorted(dirs)

        datasets = [
            [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, el))
             and not el.startswith('.')
             and any([el.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]

        return [el for el in datasets if el]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        video_name = self.videos[index]

        video = []
        transform = get_transform(self.new_size)
        for name in video_name:
            frame = Image.open(name).convert('RGB')
            frame = transform(frame)
            video.append(frame)

        return {'video': video, 'video_name': video_name}
