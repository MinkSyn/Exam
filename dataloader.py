from PIL import Image
import os
import copy
import time
import threading
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np


ROOT = os.getcwd()
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DataLoader(object):
    def __init__(self, root_path, transforms=None, batch_size=1, shuffle=True, num_workers=1):
        self.work_path = root_path
        self.transforms = None
        if not transforms is None:
            self.transforms = transforms
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.database = {}
        self.lable_package = {}
        self.init_img_parms()
        self.loader = pil_loader
        self.create_triplet_db()
        self.data_queue = []
        self.lable_queue = []
        self.data_queue_lock = threading.Lock()
        self.data_load_thread = threading.Thread(target=self.data_load)
        self.data_load_thread.start()
        self.start = 0
        self.end = math.ceil(len(self.triplet_db) / (self.batch_size * 1.0)) - 1
        self.remainder = len(self.triplet_db) % self.batch_size

    def init_img_parms(self):
        self.lables = os.listdir(self.work_path)
        self.lables.sort()
        self.lables_map = dict(zip(self.lables, range(len(self.lables))))
        for lable in self.lables:
            lable_path = os.path.join(self.work_path, lable)
            if not os.path.isdir(lable_path):
                continue
            self.lable_package[self.lables_map[lable]] = {}
            lable_imgs = os.listdir(lable_path)
            lable_imgs.sort()
            for lable_img in lable_imgs:
                if is_image_file(lable_img):
                    sample = [os.path.join(lable_path, lable_img), self.lables_map[lable]]
                    self.database["%s_%s" % (lable, lable_img)] = sample
                    self.lable_package[self.lables_map[lable]][lable_img] = sample
        self.targets = [d[1] for d in self.database.values()]

    def __getitem__(self, index):
        path, target = self.database[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, target

    def __iter__(self):
        return self

    def __next__(self):
        if self.start == 0 and len(self) == 0:
            # print ('重新开始 剩余缓冲 %d' % len(self.data_queue))
            self.create_triplet_db()
            self.data_load_thread = threading.Thread(target=self.data_load)
            self.data_load_thread.start()
        if self.start < self.end:
            # print (self.start)
            # print ('剩余缓冲 %d' % len(self.data_queue))
            self.start += 1
            while len(self.data_queue) < self.batch_size:
                time.sleep(0.2)
            with self.data_queue_lock:
                sample_tuple = (self.data_queue[:self.batch_size], self.lable_queue[:self.batch_size])
                del self.data_queue[:self.batch_size]
                del self.lable_queue[:self.batch_size]
                return sample_tuple
        elif self.start == self.end:
            # print (self.start)
            # print ('end 剩余缓冲 %d' % len(self.data_queue))
            self.start += 1
            while len(self.data_queue) < self.remainder:
                time.sleep(0.2)
            with self.data_queue_lock:
                sample_tuple = (self.data_queue[:self.batch_size], self.lable_queue[:self.batch_size])
                del self.data_queue[:self.batch_size]
                del self.lable_queue[:self.batch_size]
                return sample_tuple
        else:
            # print (self.start)
            # print ('stop 剩余缓冲 %d' % len(self.data_queue))
            self.start = 0
            raise StopIteration

    def __len__(self):
        return len(self.triplet_db)

    def create_triplet_db(self):
        self.triplet_db = []
        anchor_db = copy.deepcopy(self.lable_package)
        positive_db = copy.deepcopy(self.lable_package)
        negative_db = copy.deepcopy(self.database)

        # positive_db_keys = positive_db.keys()
        # np.random.shuffle(positive_db_keys)
        # positive_db_keys = [positive_db_keys[-1]] + positive_db_keys[:-1]
        # for positive_db_key in positive_db_keys:
        #     lable_samples = positive_db[positive_db_key]
        #     lable_sample_keys = lable_samples.keys()
        #     lable_sample_keys = [lable_sample_keys[-1]] + lable_sample_keys[:-1]

        # for key, sample in anchor_db.items():
        #     lable_i = sample[1]
        #     img_path = sample[0]
        anchor_db_keys = list(anchor_db.keys())
        if self.shuffle:
            random.shuffle(anchor_db_keys)

        for lable in anchor_db_keys:
            lable_samples_anchor = anchor_db[lable]
            lable_samples_positive = positive_db[lable]
            if len(lable_samples_anchor) == 0:
                continue
            lable_sample_anchor_keys = list(lable_samples_anchor.keys())
            if self.shuffle:
                random.shuffle(lable_sample_anchor_keys)
            if len(lable_samples_anchor) == 1:
                # lable_sample_anchor_keys = list(lable_samples_anchor.keys())
                lable_sample_positive_keys = [lable_sample_anchor_keys[-1]] + lable_sample_anchor_keys[:1]
            elif len(lable_samples_anchor) == 2:
                # print (2)
                # lable_sample_anchor_keys = list(lable_samples_anchor.keys())
                lable_sample_positive_keys = [lable_sample_anchor_keys[-1]] + lable_sample_anchor_keys[:1]

            else:
                # lable_sample_anchor_keys = list(lable_samples_anchor.keys())
                # lable_sample_positive_keys = [lable_sample_anchor_keys[-1]] + lable_sample_anchor_keys[:-1]

                mid_index = len(lable_sample_anchor_keys) // 2
                lable_sample_anchor_keys[mid_index], lable_sample_anchor_keys[-1] = lable_sample_anchor_keys[-1], \
                                                                                    lable_sample_anchor_keys[mid_index]
                lable_sample_positive_keys = lable_sample_anchor_keys[::-1]
            for key_index in range(len(lable_sample_anchor_keys)):
                sample_anchor = lable_samples_anchor[lable_sample_anchor_keys[key_index]]
                sample_positive = lable_samples_positive[lable_sample_positive_keys[key_index]]
                sample_lable = sample_anchor[1]
                while True:
                    sample_nagetive = None
                    for key, value in negative_db.items():
                        if value[1] != sample_lable:
                            sample_nagetive = negative_db[key]
                            del negative_db[key]
                            break
                    if sample_nagetive is None:
                        negative_db = copy.deepcopy(self.database)
                    else:
                        break
                triplet_sample = [sample_anchor, sample_positive, sample_nagetive]
                self.triplet_db.append(triplet_sample)
        if self.shuffle:
            random.shuffle(self.triplet_db)

    def data_load(self):
        # 既然都用了线程池和队列，起码也得放10个样本意思意思
        big_batch_pool_szie = self.batch_size * 3 if self.batch_size * 3 > 10 else 10
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        while True:
            # print ("test log: 数据池剩余:%s\t缓冲池剩余%s" % (len(self), len(self.data_queue)))
            if len(self.data_queue) > big_batch_pool_szie:
                time.sleep(0.4)
            else:
                try:
                    pop_length = self.batch_size if len(self) > self.batch_size else len(self)
                    if pop_length == 0:
                        break
                    # print ("load length: %d" % pop_length)
                    temp_samples = [self.triplet_db.pop() for i in range(pop_length)]
                    all_task = [executor.submit(self.load_sample, (sample)) for sample in temp_samples]
                    for future in as_completed(all_task):
                        data = future.result()
                except Exception as e:
                    # print ("load error: %s" % e)
                    break
        return

    def load_sample(self, sample):
        temp_imgs = []
        temp_targets = []
        for sp in sample:
            path = sp[0]
            target = sp[1]
            sample_image = self.loader(path)
            # I myself use the pytorch framework, so use the transform provided by pytorch
            # to convert the image to tensor.
            # If you don't use the pytorch framework, the method returns the normal PIL image object.
            # You can also freely change the transform you want to use,
            # just make some changes to the code inside the if.

            if self.transforms is not None:
                sample_image = self.transforms(sample_image)
                # import torch  # 调整到文件首行
                target = torch.tensor(target)
            temp_imgs.append(sample_image)
            temp_targets.append(target)
        with self.data_queue_lock:
            self.data_queue.append(temp_imgs)
            self.lable_queue.append(temp_targets)
        return True

    def check_data(self):
        for index in range(len(self.triplet_db)):
            [sample_anchor, sample_positive, sample_nagetive] = self.triplet_db[index]
            assert sample_anchor[1] == sample_positive[1]
            assert sample_anchor[1] != sample_nagetive[1]

            assert sample_anchor[0] != sample_positive[0]
            assert sample_anchor[0] != sample_nagetive[0]

        triplet_imgs = [sample[0][0] for sample in self.triplet_db]
        triplet_imgs.sort()
        base_imgs = [sample[0] for key, sample in self.database.items()]
        base_imgs.sort()
        for i in range(len(triplet_imgs)):
            assert triplet_imgs[i] == base_imgs[i]



def triplet_trans(sample: tuple):
    # target is label of [anchor, positive and negative]
    # input is data of [anchor, positive and negative]
    input, target = sample # input: 64x[3x[3x224x224]], target 64x3

    temp_batch = len(input) # Batch size

    ''' Change to tensor
    Ex  input: list([tensor([tensor(x), tensor(y)]), tensor([tensor(z), tensor(w)])])
        output: tensor([[x,y],[z,w]])'''
    temp_x = [torch.stack(input[i], dim=0) for i in range(len(input))]
    temp_y = [torch.stack(target[i], dim=0) for i in range(len(target))]

    new_x = torch.stack(temp_x, dim=0) # Input data 64x[3x[3x224x224]]
    new_y = torch.stack(temp_y, dim=0) # Labels 64x3
    
    # Change size: new_x: 64x[3x[3x224x224]] to 192x[3x224x224]
    #              new_y: 64x3 to 192
    new_x = [new_x[:, i] for i in range(3)]
    new_y = [new_y[:, i] for i in range(3)]
    # Change to tensor
    sample_input = torch.cat(new_x, 0)
    sample_target = torch.cat(new_y, 0)

    target = sample_target#.cuda(async=True)
    input_var = torch.autograd.Variable(sample_input)
    target_var = torch.autograd.Variable(target)
    
    # compute output
    anchor = input_var[:temp_batch] # Size 64x3x224x224
    positive = input_var[temp_batch:(temp_batch * 2)] # Size 64x3x224x224
    negative = input_var[-temp_batch:]

    return anchor, positive, negative, target_var


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.CenterCrop(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    count = 0
    batch_size = 64
    train_loader = DataLoader(root_path=ROOT+'/Train',
                                  batch_size=batch_size, num_workers=8, transforms=transform)

    epochs = 1
    for epoch in range(epochs):

        print ('=' * 20)
        
        for idx, sample in enumerate(train_loader):
            try:
                print ('load data %d' % (idx))
                anchor, positive, negative, labels = triplet_trans(sample)

                assert anchor.size() == positive.size()
                assert anchor.size() == negative.size()
            except Exception as e:
                print ('input len :%s' % len(input))
                print ('target len :%s' % len(target))
                print ('new_x len :%s' % len(new_x))
                print ('new_y len :%s' % len(new_y))
                print ('sample_input size :')
                print (sample_input.size())
                print ('sample_target size :')
                print (sample_target.size())
                print (anchor.size())
                print (positive.size())
                print (negative.size())