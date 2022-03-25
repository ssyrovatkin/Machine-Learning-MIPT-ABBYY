import torch
from torch.utils.data import Dataset
import gzip
import numpy as np
import torch
import matplotlib.pyplot as plt

class MNISTDataset(Dataset):
    def __init__(self, image_data_root, label_data_root):

        self.image_data_root = image_data_root
        self.image_magic_number = 0
        self.num_images = 0
        self.image_rows = 0
        self.image_columns = 0
        self.images = np.empty(0)

        self.label_data_root = label_data_root
        self.label_magic_number = 0
        self.num_labels = 0
        self.labels = np.empty(0)

        self.image_init_dataset()
        self.label_init_dataset()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def image_init_dataset(self):
        image_file = gzip.open(self.image_data_root, 'r')

        reorder_type = np.dtype(np.int32).newbyteorder('>')

        self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]

        buffer = image_file.read(self.num_images * self.image_rows * self.image_columns)

        self.images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        self.images = np.reshape(self.images, (self.num_images, 1, 28, 28))
        self.images = self.images / 255
        self.images = torch.tensor(self.images)

    def label_init_dataset(self):
        label_file = gzip.open(self.label_data_root, 'r')

        reorder_type = np.dtype(np.int32).newbyteorder('>')
        self.label_magic_number = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        self.num_labels = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]

        buffer = label_file.read(self.num_labels)
        self.labels = np.frombuffer(buffer, dtype=np.uint8)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

def draw_image(images_root, labels_root, image_idx):
    mnist = MNISTDataset(images_root, labels_root)

    mnist.images = np.reshape(mnist.images, (mnist.num_images, 28, 28))
    image, label = mnist.__getitem__(image_idx)
    print('Image dimensions: {}x{}'.format(image.shape[0], image.shape[1]))
    print('Label: {}'.format(label.item()))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":

    draw_image('C:/Users/mnist/train-images-idx3-ubyte.gz', 'C:/Users/mnist/train-labels-idx1-ubyte.gz', 500)
    draw_image('C:/Users/mnist/t10k-images-idx3-ubyte.gz', 'C:/Users/mnist/t10k-labels-idx1-ubyte.gz', 500)
