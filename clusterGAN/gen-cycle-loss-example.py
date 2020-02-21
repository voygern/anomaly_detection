from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import sys

    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt

    import pandas as pd

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image

    from itertools import chain as ichain

    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.models import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.utils import sample_z
    from clusgan.datasets import get_dataloader, dataset_list

    from sklearn.manifold import TSNE
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Script to save generated examples from learned ClusterGAN generator")
    parser.add_argument("-r", "--run_dir", dest="run_dir", help="Training run directory")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=100, type=int, help="Batch size")
    args = parser.parse_args()

    batch_size = args.batch_size

    # Directory structure for this run
    run_dir = args.run_dir.rstrip("/")
    run_name = run_dir.split(os.sep)[-1]
    dataset_name = run_dir.split(os.sep)[-2]

    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    # Latent space info
    train_df = pd.read_csv('%s/training_details.csv' % (run_dir))
    latent_dim = train_df['latent_dim'][0]
    n_c = train_df['n_classes'][0]

    cuda = True if torch.cuda.is_available() else False

    # Load encoder model
    encoder = Encoder_CNN(latent_dim, n_c)
    enc_fname = os.path.join(models_dir, encoder.name + '.pth.tar')
    encoder.load_state_dict(torch.load(enc_fname))
    if cuda:
        encoder.cuda()
    encoder.eval()

    # Load generator model
    x_shape = (1, 28, 28)
    generator = Generator_CNN(latent_dim, n_c, x_shape)
    gen_fname = os.path.join(models_dir, generator.name + '.pth.tar')
    generator.load_state_dict(torch.load(gen_fname))
    if cuda:
        generator.cuda()
    generator.eval()

    # Loop through specific classes
    # for idx in range(n_c):
    #     zn, zc, zc_idx = sample_z(shape=batch_size, latent_dim=latent_dim, n_c=n_c, fix_class=idx, req_grad=False)
    #
    #     # Generate a batch of images
    #     gen_imgs = generator(zn, zc)
    #
    #     # Save some examples!
    #     save_image(gen_imgs.data, '%s/class_%i_gen.png' % (imgs_dir, idx),
    #                nrow=int(np.sqrt(batch_size)), normalize=True)
    #
    #     enc_zn, enc_zc, enc_zc_logits = encoder(gen_imgs)
    #
    #     # Generate a batch of images
    #     gen_imgs = generator(enc_zn, enc_zc)
    #
    #     # Save some examples!
    #     save_image(gen_imgs.data, '%s/class_enc_%i_gen.png' % (imgs_dir, idx),
    #                nrow=int(np.sqrt(batch_size)), normalize=True)
    #     enc_zn, enc_zc, enc_zc_logits = encoder(gen_imgs)

    test_batch_size = 64
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    testdata = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=test_batch_size, train_set=False
                              , shuffle=False)
    test_imgs, test_labels = next(iter(testdata))
    # test_imgs = Variable(test_imgs.type(Tensor))
    test_imgs = test_imgs[test_labels==1]
    stack_imgs = []

    # for t in test_imgs:
    #     zn,zc,zc_logit = encoder(t.unsqueeze(0))
    #     gen_pic = generator(zn,zc)
    #     print(zc.data)
    #     if (len(stack_imgs) == 0):
    #         stack_imgs = gen_pic
    #     else:
    #         stack_imgs = torch.cat((stack_imgs, gen_pic), 0)
    zn,zc,zc_logit = encoder(test_imgs)
    stack_imgs = generator(zn,zc)
    print(zc.argmax(1))
    save_image(stack_imgs,
               '%s/agen_test_%i.png' % (imgs_dir, 1),
               nrow=8, normalize=True)
    save_image(test_imgs,
               '%s/agen_test_o%i.png' % (imgs_dir, 1),
               nrow=8, normalize=True)

if __name__ == "__main__":
    main()
