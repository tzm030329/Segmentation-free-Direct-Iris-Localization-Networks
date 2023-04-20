#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################
# filename: test.py
# datetime: 2023-04-12 17:07:51
######################


import numpy as np
import cv2
import torch
from glob import glob
from matplotlib import pyplot as plt

import models


def main():
    pass


if __name__ == '__main__':
    main()

    debug = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    # Change here for your evaluation
    files = sorted(glob('img/sample/**/*.png', recursive=True))


    # define model
    model = models.ILN()
    state_dict = torch.load('pth/casia-thousand_ILN.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    files = files[:1] if debug else files

    print('filepath,px,py,pr,ix,iy,ir')
    for i, ifile in enumerate(files):

        # read image file
        img = cv2.imread(ifile, cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != (480, 640):
            # image size must be VGA (480x640)
            print('resize image to (480, 640)')
            img = cv2.resize(img, (640,480), interpolation=cv2.INTER_CUBIC)

        # transform img into torch tensor
        x = img/255.
        h,w = x.shape
        x = x.reshape(1,1,h,w)
        x = torch.from_numpy(x.astype(np.float32))
        x = x.to(device)

        # predict
        y = model.forward(x)
        y = y.cpu().detach().numpy().flatten()

        # output results
        outstr = '%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (ifile, y[0],y[1],y[2],y[3],y[4],y[5])
        print(outstr)

        # to array
        y = y.reshape(1,-1)
        ys = y if i==0 else np.append(ys, y, axis=0)

        if debug:
            # create circle iamge
            px, py, pr, ix, iy, ir = np.round(y).astype(np.int32)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.circle(img, (ix, iy), ir, (255,255,0), thickness=2)
            img = cv2.circle(img, (px, py), pr, (255,  0,0), thickness=2)


    outputs = np.column_stack((files, ys))
    if not debug:
        np.savetxt('results.csv', outputs, fmt='%s', delimiter=',')

    if debug:
        plt.imshow(img)
        plt.show()
