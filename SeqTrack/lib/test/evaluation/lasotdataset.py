import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LaSOTDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasot_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['elephant-1',
                         'elephant-12',
                         'elephant-16',
                         'elephant-18',
                         'goldfish-3',
                         'goldfish-7',
                         'goldfish-8',
                         'goldfish-10',
                         'flag-3',
                         'flag-9',
                         'flag-5',
                         'flag-2',
                         'frog-3',
                         'frog-4',
                         'frog-20',
                         'frog-9',
                         'gametarget-1',
                         'gametarget-2',
                         'gametarget-7',
                         'gametarget-13',
                         'electricfan-1',
                         'electricfan-10',
                         'electricfan-18',
                         'electricfan-20',
                         'gecko-1',
                         'gecko-5',
                         'gecko-16',
                         'gecko-19',
                         'fox-2',
                         'fox-3',
                         'fox-5',
                         'fox-20',
                         'giraffe-2',
                         'giraffe-10',
                         'giraffe-13',
                         'giraffe-15',
                         'gorilla-4',
                         'gorilla-6',
                         'gorilla-9',
                         'gorilla-13',
                        ]
        return sequence_list
