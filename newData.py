import numpy as np
from scipy.io import loadmat
from scipy.io import savemat



def reverseData(inp, outp):
    img_ori = loadmat(inp)['X'].transpose(3,0,1,2)
    la_ori = loadmat(inp)['y']
    o_im = []
    o_la = []
    n,w,h,c  = img_ori.shape
    print(img_ori.shape,  la_ori.shape)
    for i in range(n):
        img_rev = 255 - img_ori[i]
        o_im.append(img_rev)
        o_la.append(la_ori[i])
    o_im, o_la = np.array(o_im).transpose(1,2,3,0), np.array(o_la)
    print(o_im.shape,  o_la.shape)
    savemat(outp, {'X':o_im, 'y':o_la})

def subSet(inp, outp, n):
    ori = loadmat(inp)
    img = ori['X']
    la = ori['y']
    o_im = img[:, :,:, 0:n]
    o_la = la[0:n]
    print(o_im.shape, o_la.shape)
    savemat(outp, {'X': o_im, 'y': o_la})


def bigSmall():
    path = './dataset/train_32x32.mat'
    ori = loadmat(path)['X'].transpose(3,0,1,2)
    la = loadmat(path)['y']
    o_im, o_la = [], []
    cnt_1, cnt_2, cnt_3, cnt_4, cnt_5 = 0,0,0,0,0
    print(ori.shape)
    n, w,h,c=  ori.shape
    randl = np.arange(int(n))
    np.random.shuffle(randl)
    for i in randl:
        if la[i] == 6:
            #if cnt_1 >= 1000:
                #continue
            cnt_1+=1
        if la[i] == 7:
            #if cnt_2 >= 1000:
                #continue
            cnt_2+=1
        if la[i] == 8:
            if cnt_3 >= 500:
                continue
            cnt_3+=1
        if la[i] == 9:
            if cnt_4 >= 500:
                continue
            cnt_4+=1
        if la[i] == 0:
            if cnt_5 >= 500:
                continue
            cnt_5+=1
        o_la.append(la[i])
        o_im.append(ori[i])
    o_im, o_la = np.array(o_im).transpose(1,2,3,0), np.array(o_la)
    print(o_im.shape, o_la.shape)
    savemat('./dataset/train_890_32x32.mat', {'X':o_im, 'y':o_la})

if __name__ == "__main__":
    subSet('./dataset/train_12345_32x32.mat', './dataset/train_12345_10000_32x32.mat', 10000)
    subSet('./dataset/train_67890_32x32.mat', './dataset/train_67890_10000_32x32.mat', 10000)
    subSet('./dataset/train_890_32x32.mat', './dataset/train_890_10000_32x32.mat', 10000)
    subSet('./dataset/train_12345_32x32.mat', './dataset/train_12345_30000_32x32.mat', 30000)
    subSet('./dataset/train_67890_32x32.mat', './dataset/train_67890_30000_32x32.mat', 30000)
    subSet('./dataset/train_890_32x32.mat', './dataset/train_890_30000_32x32.mat', 30000)
    subSet('./dataset/train_32x32.mat', './dataset/train_10000_32x32.mat', 10000)
    subSet('./dataset/train_32x32.mat', './dataset/train_30000_32x32.mat', 30000)
    reverseData('./dataset/train_32x32.mat', './dataset/train_rev_32x32.mat')
    reverseData('./dataset/train_10000_32x32.mat', './dataset/train_rev_10000_32x32.mat')
    reverseData('./dataset/train_30000_32x32.mat', './dataset/train_rev_30000_32x32.mat')
