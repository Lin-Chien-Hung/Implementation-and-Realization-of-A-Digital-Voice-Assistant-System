import numpy as np
import random

import torch
import torchvision.transforms as transforms

from generator.SR_Dataset import ToTensorInput
import configure as c

class metaGenerator(object):

    def __init__(self, test_DB, enroll_DB, file_loader, enroll_length, test_length,
                 nb_classes=3, n_support=1, n_query=1, max_iter=100, xp=np):
        super(metaGenerator, self).__init__()

        self.nb_classes = nb_classes
        self.n_support = n_support
        self.n_query = n_query
        self.nb_samples_per_class = n_support+ n_query

        self.enroll_length = enroll_length
        self.test_length = test_length

        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.test_data = self._load_data(test_DB)
        self.enroll_data = self._load_data(enroll_DB)
        self.file_loader = file_loader
        self.transform = transforms.Compose([
            ToTensorInput()  # torch tensor:(1, n_dims, n_frames)
        ])

    def _load_data(self, data_DB):
        nb_speaker = len(set(data_DB['labels']))

        return {key: np.array(data_DB.loc[data_DB['labels']==key]['filename']) for key in range(nb_speaker)}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()

    def cut_frames(self, frames_features, mode='enroll'):
        # Normalizing before slicing
        network_inputs = []
        num_frames = len(frames_features)

        if mode == 'enroll': win_size = self.enroll_length
        elif mode == 'test': win_size = self.test_length

        half_win_size = int(win_size / 2)
        
        # 如果測試語句不滿設定語句長度，一直重複測試語句直到超過為止
        while num_frames <= win_size: 
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)


        j = random.randrange(half_win_size, num_frames - half_win_size)
        if not j:
            frames_slice = np.zeros(num_frames, c.FILTER_BANK, 'float64')
            frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
        else:
            frames_slice = frames_features[j - half_win_size:j + half_win_size]
        network_inputs.append(frames_slice)

        return np.array(network_inputs)   

    def sample(self, nb_classes, nb_samples_per_class):
        
        picture_list = sorted(set(self.enroll_data.keys()))

        #這個sample_classes就是看你test資料夾有個個語者 ex: 如果有10個 那sample_classes 就是[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #nb_classes 是設定的註冊語者個數
        #sample_classes可大於nb_classes
        #這裡將sample_classes裡面的資料固定前nb_classes個為註冊語者
        #當sample_classes = nb_classes就代表裡面的資料都為註冊語者

        sample_classes = list(self.enroll_data.keys())
        #sample_classes = random.sample(self.test_data.keys(), nb_classes)

        labels_and_images = []
        double_check = []

        for (k, char) in enumerate(sample_classes):
            label = picture_list[char] #nb_samples_per_class = 2, 1 for support, 1 for query.
            
            en_data = self.enroll_data[char]
            te_data = self.test_data[0]
            # sample support
            labels_and_images.extend([(label, self.transform(self.cut_frames(self.file_loader(en_data[i]), mode='enroll'))) for i in range(self.n_support)])
            # sample query
            labels_and_images.extend([(label, self.transform(self.cut_frames(self.file_loader(te_data[0]), mode='test')))])
            # sample query (double_check)
            double_check.extend([(label, self.transform(self.cut_frames(self.file_loader(te_data[0]), mode='test')))])

            #labels_and_images為[(0,[a0]), (0, [b0]), (0, [c0]), (0, [d0]), (0, [e0]), (0, [f0]), (1, [a0]), (1, [b0]) .....] 其中a為enroll, 剩下b c d e f為測試語料
            #前面那個數字代表label, 有幾個註冊語者(nb_class_test) label就會到幾
            
        arg_labels_and_images = []
        for i in range(nb_samples_per_class):
            for j in range(nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])    

        #arg_labels_and_images為[(0,[s0] ), (1, [s1]), (2, [s2]), (0,[q0] ), (1, [q1]), (2, [q2])] 前面label表示語者，s,q代表support, query
        labels, images = zip(*arg_labels_and_images) 
        b_labels, b_images = zip(*double_check) 
        #print("double_check: ",double_check.size())
        


        support = torch.stack(images[:self.n_support * self.nb_classes], dim=0)  #torch.Size([3, 1, 40, 150])
        query = torch.stack(images[self.n_support*self.nb_classes:], dim=0)    #torch.Size([3, 1, 40, 500])
        b_query =  torch.stack(b_images, dim=0)  #continue  -------------

        labels = torch.tensor(labels, dtype=torch.long)

        return (support, query, b_query), labels

