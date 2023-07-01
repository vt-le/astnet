import os
import glob
import numpy as np
import scipy.io as scio


class Label:
    def __init__(self, config):
        root = config.DATASET.ROOT
        dataset_name = config.DATASET.DATASET
        if dataset_name == 'shanghaitech':
            self.frame_mask = os.path.join(root, dataset_name, 'test_frame_mask/*')
        mat_name = dataset_name + '.mat'

        self.mat_path = os.path.join(root, dataset_name, mat_name)

        test_set = config.DATASET.TESTSET
        test_dataset_path = os.path.join(root, dataset_name, test_set)
        video_folders = (os.listdir(test_dataset_path))
        video_folders.sort()
        self.video_folders = [os.path.join(test_dataset_path, folder) for folder in video_folders]
        self.dataset_name = dataset_name

    def __call__(self):
        if self.dataset_name == 'shanghaitech':
            np_list = glob.glob(self.frame_mask)
            np_list.sort()

            gt = []
            for npy in np_list:
                gt.append(np.load(npy))

            return gt
        else:
            abnormal_mat = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

            all_gt = []
            for i in range(abnormal_mat.shape[0]):
                length = len(os.listdir(self.video_folders[i]))
                sub_video_gt = np.zeros((length,), dtype=np.int8)

                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1   # TODO
                    end = one_abnormal[1, j]

                    sub_video_gt[start: end] = 1

                all_gt.append(sub_video_gt)

            return all_gt
