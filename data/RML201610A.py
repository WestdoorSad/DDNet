import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchnet as tnt

RML201610A_EVAL_CLASSES = ('PAM4', 'QAM16', 'CPFSK', '8PSK')


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

from scipy.signal import stft, get_window, istft
import librosa
def get_stft(iq_signal, fs=200000, nperseg = 128 // 4):

    noverlap = nperseg - 4  # 重叠部分长度
    frequencies, times, stft_result_complex = stft(iq_signal[0] + 1j * iq_signal[1],
                                                   fs=fs, nfft=32, return_onesided=False,
                                                   window='hann', nperseg=nperseg, noverlap=noverlap)
    stft_real = stft_result_complex.real
    stft_imag = stft_result_complex.imag
    stft_data = np.stack([stft_real, stft_imag]).astype(np.float32)
    stft_data = stft_data[:, :, :-1]
    stft_data = np.abs(stft_data)
    stft_data = librosa.amplitude_to_db(stft_data)

    return stft_data

def normalize_numpy(X):
    stfts = []
    for i in range(X.shape[0]):
        stfts.append(get_stft(X[i]))

    stfts = np.stack(stfts)

    return stfts

def advance_iq(iq):
    iq = iq[:, ::-1]
    return iq


class Pretrain_RML201610A_Dataset(data.Dataset):
    def __init__(self):
        self.name = 'RML201610A_pretrain'

        dataset_path = r"D:\信号调制\RML2016.10a_dict.pkl"
        Xd = pickle.load(open(dataset_path, 'rb'), encoding='iso-8859-1')
        all_mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
        self.eval_cls_index = [all_mods.index(cls) for cls in RML201610A_EVAL_CLASSES]

        snrs = [i for i in range(-10, 20, 2)]
        self.data = []
        self.stft = []
        self.labels = []
        self.lbl = []
        self.tomod = []

        ind = 0
        for i, mod in enumerate(all_mods):
            mod_allx = []

            for snr in snrs:
                data = Xd[(mod, snr)]
                mod_allx.append(data)
                self.lbl.extend([snr] * len(data))
                self.tomod.extend([mod] * len(data))
            mod_allx = np.concatenate(mod_allx, axis=0)
            self.data.append(np.arange(0, len(mod_allx)) + ind)
            ind += len(mod_allx)
            mod_allx = mod_allx[:, np.newaxis]
            self.stft.extend(mod_allx)
            self.labels.extend([i] * len(mod_allx))

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)



    def __getitem__(self, index):
        img = self.stft[index]
        img = torch.FloatTensor(img)

        if (index + 1) % 1000 == 0:
            img_next = self.stft[index]
            img_next = torch.FloatTensor(img_next)
        else:
            img_next = self.stft[index + 1]
            img_next = torch.FloatTensor(img_next)

        radio_start = index // 1000 * 1000
        if radio_start == 0:
            random_select_before = np.random.randint(radio_start + 1000, len(self.stft))
        else:
            random_select_before = np.random.randint(0, radio_start)
        if radio_start + 1000 >= len(self.stft):
            random_select_after = np.random.randint(0, radio_start)
        else:
            random_select_after = np.random.randint(radio_start + 1000, len(self.stft))
        random_select = np.random.choice([random_select_before, random_select_after])
        img_random = self.stft[random_select]
        img_random = torch.FloatTensor(img_random)
        label = self.labels[index]
        if label in self.eval_cls_index:
            label = -1

        return img, img_next, img_random, label

    def __len__(self):
        return len(self.labels)

class RML201610A_Dataset(data.Dataset):
    def __init__(self, phase='train', startidx=0):
        self.phase = phase
        self.name = 'RML201610A_' + phase

        dataset_path = r"D:\信号调制\RML2016.10a_dict.pkl"
        Xd = pickle.load(open(dataset_path, 'rb'), encoding='iso-8859-1')
        all_mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
        self.eval_cls_index = [all_mods.index(cls) for cls in RML201610A_EVAL_CLASSES]
        self.train_cls_index = [ind for ind in range(len(all_mods)) if ind not in self.eval_cls_index]

        # ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        if phase == 'train':
            snrs = [i for i in range(-10, 20, 2)]
        else:
            snrs = [i for i in range(-10, 20, 2)]


        self.data = []
        self.stft = []
        self.labels = []
        self.lbl = []
        self.tomod = []
        self.startidx = startidx

        ind = 0
        for i, mod in enumerate(all_mods):
            mod_allx = []
            if phase != "train":
                if i in self.train_cls_index:
                    continue
            else:
                if i in self.eval_cls_index:
                    continue

            # if phase == "train" and i not in [6, 1, 8, 3, 4, 5]:
            #     continue
            for snr in snrs:
                data = Xd[(mod, snr)]
                # if phase == "train" and i not in [6, 1, 8, 3, 4, 5]:
                #     select_ind = np.random.choice(range(len(data)), size=int(len(data) * 0.001), replace=False)
                #     data = data[select_ind]
                mod_allx.append(data)
                self.lbl.extend([snr] * len(data))
                self.tomod.extend([mod] * len(data))
            mod_allx = np.concatenate(mod_allx, axis=0)
            self.data.append(np.arange(0, len(mod_allx)) + ind)
            ind += len(mod_allx)
            mod_allx = mod_allx[:, np.newaxis]
            # stft_allx = normalize_numpy(mod_allx)
            self.stft.extend(mod_allx)
            self.labels.extend([i + self.startidx] * len(mod_allx))

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)



        self.labelIds_base = list(buildLabelIndex(self.labels).keys())
        self.labelIds_novel = self.eval_cls_index
        self.num_cats_base = len(self.labelIds_base)
        self.num_cats_novel = len(self.labelIds_novel)



    def __getitem__(self, index):
        img, label = self.stft[index], self.labels[index]
        img = torch.FloatTensor(img)
        label = torch.LongTensor(label)
        return img, label

    def __len__(self):
        return len(self.labels)

class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel <= max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def load_function(self, iter_idx):
        Exemplars, Test, Kall, nKbase = self.sample_episode()
        Xt, Yt = self.createExamplesTensorData(Test)
        Kall = torch.LongTensor(Kall)
        LBe = torch.LongTensor([self.dataset.lbl[img_idx] for img_idx, _ in Exemplars])
        LBt = torch.LongTensor([self.dataset.lbl[img_idx] for img_idx, _ in Test])

        if len(Exemplars) > 0:
            Xe, Ye = self.createExamplesTensorData(Exemplars)
            return Xe, Ye, Xt, Yt, Kall, nKbase, LBe, LBt
        else:
            return Xt, Yt, Kall, nKbase

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)


        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=self.load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
