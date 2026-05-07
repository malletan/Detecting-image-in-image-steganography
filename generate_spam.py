from math import sqrt, pi
import numpy as np
from scipy.signal import convolve, convolve2d
from scipy.fftpack import idct

from PIL import Image
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
import os

from tqdm import tqdm


def mask_singles(T, mask_func, mask):
    m2 = np.ones((2*T+1,2*T+1,2*T+1,2*T+1), int)
    a = mask_func(m2,mask)
    return np.where(a != 0)


def takeSingles(m, mask):
    m = np.asanyarray(m)
    return np.where(mask,m,np.zeros(1, m.dtype))


def locate_singles_bins_spam(T):
    np.random.seed(0)
    tabRand = np.random.randn(2*T+1,2*T+1,2*T+1,2*T+1)
    tabSimSign = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    tabSimScan = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    tabTaken = np.ones((2*T+1,2*T+1,2*T+1,2*T+1))
    for d1 in range(-T,T+1):
        for d2 in range(-T,T+1):
            for d3 in range(-T,T+1):
                for d4 in range(-T,T+1):
                    tabSimSign[d1+T,d2+T,d3+T,d4+T] = tabRand[-d1+T,-d2+T,-d3+T,-d4+T]
    tabSimSign += tabRand

    for d1 in range(-T,T+1):
        for d2 in range(-T,T+1):
            for d3 in range(-T,T+1):
                for d4 in range(-T,T+1):
                    tabSimScan[d1+T,d2+T,d3+T,d4+T] = tabSimSign[d4+T,d3+T,d2+T,d1+T]
    tabSimScan += tabSimSign
    
    a , indTaken = np.unique(tabSimScan, return_index=True)    
    
    tabTaken = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    tabTaken.flat[indTaken]=1
    out_log = np.logical_and(tabTaken,np.ones((2*T+1,2*T+1,2*T+1,2*T+1)))
    out = mask_singles(T, takeSingles, out_log)
    return out


def locate_singles_bins_minmax(T):
    np.random.seed(0)
    tabRand1 = np.random.randn(2*T+1,2*T+1,2*T+1,2*T+1)
    tabRand2 = np.random.randn(2*T+1,2*T+1,2*T+1,2*T+1)
    tabSimSign = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    tabSimScan = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    tabTaken = np.ones((2*T+1,2*T+1,2*T+1,2*T+1))
    
    for d1 in range(-T,T+1):
        for d2 in range(-T,T+1):
            for d3 in range(-T,T+1):
                for d4 in range(-T,T+1):
                    tabSimSign[d1+T,d2+T,d3+T,d4+T] = tabRand1[-d1+T,-d2+T,-d3+T,-d4+T]
    tabSimSign += tabRand2

    for d1 in range(-T,T+1):
        for d2 in range(-T,T+1):
            for d3 in range(-T,T+1):
                for d4 in range(-T,T+1):
                    tabSimScan[d1+T,d2+T,d3+T,d4+T] = tabSimSign[d4+T,d3+T,d2+T,d1+T]
    tabSimScan += tabSimSign
    
    a , indTaken = np.unique(tabSimScan, return_index=True)    
    
    tabTaken = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    tabTaken.flat[indTaken]=1
    out_log = np.logical_and(tabTaken,np.ones((2*T+1,2*T+1,2*T+1,2*T+1)))
    out = mask_singles(T, takeSingles, out_log)
    return out


def Quant(res, q, T):
    if(q!=1):
        res_quant = res_quant = np.round((res+0.01*np.sign(res))/q)
        res_quant = res_quant.astype(np.int16)
    else:
        res_quant = res.astype(np.int16)

    res_quant[res_quant>=T]=T
    res_quant[res_quant<=-T]=-T
    res_quant = res_quant+T
    return res_quant


def Cooc(res, dir, T):
    nbins = 2*T+1
    M,N = res.shape
    flatcount = np.zeros(nbins**4)
    if dir == 'hor':
        xassign = (res[:,0:N-3]).flatten()
        yassign = (res[:,1:N-2]).flatten()
        zassign = (res[:,2:N-1]).flatten()
        wassign = (res[:,3:N-0]).flatten()
        flatcount = np.bincount(xassign + yassign * nbins + zassign * nbins**2 + wassign * nbins**3, minlength=nbins**4)
    elif dir == 'ver':
        xassign = (res[0:M-3,:]).flatten()
        yassign = (res[1:M-2,:]).flatten()
        zassign = (res[2:M-1,:]).flatten()
        wassign = (res[3:M-0,:]).flatten()
        flatcount = np.bincount(xassign + yassign * nbins + zassign * nbins**2 + wassign * nbins**3, minlength=nbins**4)

    out = flatcount.reshape((nbins, nbins, nbins, nbins)).T
    out = out/float(np.sum(out))

    return out


def sym_spam(mat_cooc, T):
    mat_sym = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    mat_sym_out = np.asarray(mat_cooc,dtype=float)
    # Perform a central symmetry
    for x in range(2 * T + 1):
        for y in range(2 * T + 1):
            for z in range(2 * T + 1):
                for w in range(2 * T + 1):
                    mat_sym[2 * T - x, 2 * T - y, 2 * T - z, 2 * T - w] = mat_cooc[x,y,z,w]
    mat_sym_out += mat_sym

    # Perform a scan symmetry
    mat_sym_2 = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    for x in range(2 * T + 1):
        for y in range(2 * T + 1):
            for z in range(2 * T + 1):
                for w in range(2 * T + 1):
                    mat_sym_2[w,z,y,x] = mat_sym_out[x,y,z,w]
    mat_sym_out += mat_sym_2
    
    return mat_sym_out
    

def sym_minmax(mat_cooc_min, mat_cooc_max, T):

    mat_sym_max = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    mat_sym_out = np.asarray(mat_cooc_min,dtype=float)
    # Perform a central symmetry involving min and max
    for x in range(2 * T + 1):
        for y in range(2 * T + 1):
            for z in range(2 * T + 1):
                for w in range(2 * T + 1):
                    mat_sym_max[2 * T - x, 2 * T - y, 2 * T - z, 2 * T - w] = mat_cooc_max[x,y,z,w]
    mat_sym_out += mat_sym_max

    # Perform a scan symmetry
    mat_sym_2 = np.zeros((2*T+1,2*T+1,2*T+1,2*T+1))
    for x in range(2 * T + 1):
        for y in range(2 * T + 1):
            for z in range(2 * T + 1):
                for w in range(2 * T + 1):
                    mat_sym_2[w,z,y,x] = mat_sym_out[x,y,z,w]
    mat_sym_out += mat_sym_2
    
    return mat_sym_out


def compute_SPAM(img_path):
    _T = 2

    mask_spam = locate_singles_bins_spam(_T)
    mask_minmax = locate_singles_bins_minmax(_T)

    # Open the image
    X = np.array(Image.open(img_path).convert("L"))
    T= _T

    X = X.astype(np.int16)
    M,N = X.shape

    # Residual declaration
    R1 = np.zeros([M,N])
    L1 = np.zeros([M,N])
    D1 = np.zeros([M,N])
    U1 = np.zeros([M,N])
    RU1 = np.zeros([M,N])
    RD1 = np.zeros([M,N])
    LU1 = np.zeros([M,N])
    LD1 = np.zeros([M,N])
    H2 = np.zeros([M,N])
    V2 = np.zeros([M,N])
    D2 = np.zeros([M,N])
    M2 = np.zeros([M,N])
    U3 = np.zeros([M,N])
    D3 = np.zeros([M,N])
    R3 = np.zeros([M,N])
    L3 = np.zeros([M,N])
    RU3 = np.zeros([M,N])
    RD3 = np.zeros([M,N])
    LU3 = np.zeros([M,N])
    LD3 = np.zeros([M,N])
    EdgeU3 = np.zeros([M,N])
    EdgeD3 = np.zeros([M,N])
    EdgeR3 = np.zeros([M,N])
    EdgeL3 = np.zeros([M,N])
    Square3 = np.zeros([M,N])
    Square5 = np.zeros([M,N])
    EdgeU5 = np.zeros([M,N])
    EdgeD5 = np.zeros([M,N])
    EdgeR5 = np.zeros([M,N])
    EdgeL5 = np.zeros([M,N])

    # Note: in order to save time, we try to minimize the number of matrix operations

    # First order
    R1[:,0:N-1] = X[:,1:N] - X[:,0:N-1]
    L1[:,1:N] = -R1[:,0:N-1]
    D1[0:M-1,:] = X[1:M,:]- X[0:M-1,:]
    U1[1:M,:] = -D1[0:M-1,:]
    RD1[0:M-1,0:N-1] = X[1:M,1:N] - X[0:M-1,0:N-1]
    LU1[1:M,1:N] = -RD1[0:M-1,0:N-1]
    RU1[1:M,0:N-1] = X[0:M-1,1:N] - X[1:M,0:N-1]
    LD1[0:M-1,1:N] = -RU1[1:M,0:N-1]

    # Second order
    H2[:,1:N-1] = R1[:,1:N-1] + L1[:,1:N-1]
    V2[1:M-1,:] = U1[1:M-1,:] + D1[1:M-1,:]
    D2[1:M-1,1:N-1] = LU1[1:M-1,1:N-1] + RD1[1:M-1,1:N-1]
    M2[1:M-1,1:N-1] = LD1[1:M-1,1:N-1] + RU1[1:M-1,1:N-1]

    # Third order
    U3[2:M-1,:] = V2[2:M-1,:] - V2[1:M-2,:]
    D3[1:M-2,:] = V2[1:M-2,:] - V2[2:M-1,:]
    R3[:,1:N-2] = H2[:,1:N-2] - H2[:,2:N-1]
    L3[:,2:N-1] = H2[:,2:N-1] - H2[:,1:N-2]  
    RU3[2:M-1,1:N-2] = M2[2:M-1,1:N-2] - M2[1:M-2,2:N-1]
    LD3[1:M-2,2:N-1] = M2[1:M-2,2:N-1] - M2[2:M-1,1:N-2]
    RD3[1:M-2,1:N-2] = D2[1:M-2,1:N-2] - D2[2:M-1,2:N-1]
    LU3[2:M-1,2:N-1] = D2[2:M-1,2:N-1] - D2[1:M-2,1:N-2]

    # Edge 3x3: 
    EdgeU3[1:M,1:N-1] = 2*H2[1:M,1:N-1] - H2[0:M-1,1:N-1]
    EdgeD3[0:M-1,1:N-1] = 2*H2[0:M-1,1:N-1] - H2[1:M,1:N-1]
    EdgeR3[1:M-1,0:N-1] = 2*V2[1:M-1,0:N-1] - V2[1:M-1,1:N]
    EdgeL3[1:M-1,1:N] = 2*V2[1:M-1,1:N] - V2[1:M-1,0:N-1]
    Square3[1:M-1,1:N-1] = 2*H2[1:M-1,1:N-1] - H2[0:M-2,1:N-1] - H2[2:M,1:N-1]

    # Edge 5x5
    Square5[2:M-2,2:N-2] = 2*Square3[2:M-2,2:N-2] + Square3[3:M-1,3:N-1] + Square3[3:M-1,1:N-3] + Square3[1:M-3,3:N-1] + Square3[1:M-3,1:N-3]
    EdgeU5[2:M-2,2:N-2] = Square5[2:M-2,2:N-2] - EdgeD3[3:M-1,3:N-1] - EdgeD3[3:M-1,1:N-3] + 2*H2[3:M-1,2:N-2]
    EdgeD5[2:M-2,2:N-2] = Square5[2:M-2,2:N-2] - EdgeU3[1:M-3,3:N-1] - EdgeU3[1:M-3,1:N-3] + 2*H2[1:M-3,2:N-2]
    EdgeR5[2:M-2,2:N-2] = Square5[2:M-2,2:N-2] - EdgeL3[3:M-1,1:N-3] - EdgeL3[1:M-3,1:N-3] + 2*V2[2:M-2,1:N-3]
    EdgeL5[2:M-2,2:N-2] = Square5[2:M-2,2:N-2] - EdgeR3[3:M-1,3:N-1] - EdgeR3[1:M-3,3:N-1] + 2*V2[2:M-2,3:N-1]

    #We remove the resildual borders that are inconsistent
    R1 = R1[1:M-1,1:N-1]
    L1 = L1[1:M-1,1:N-1]
    D1 = D1[1:M-1,1:N-1]
    U1 = U1[1:M-1,1:N-1]
    RU1 = RU1[1:M-1,1:N-1]
    LD1 = LD1[1:M-1,1:N-1]
    RD1 = RD1[1:M-1,1:N-1]
    LU1 = LU1[1:M-1,1:N-1]

    H2 = H2[1:M-1,1:N-1]
    V2 = V2[1:M-1,1:N-1]
    D2 = D2[1:M-1,1:N-1]
    M2 = M2[1:M-1,1:N-1]

    R3 = R3[2:M-2,2:N-2]
    L3 = L3[2:M-2,2:N-2]
    D3 = D3[2:M-2,2:N-2]
    U3 = U3[2:M-2,2:N-2]
    RU3 = RU3[2:M-2,2:N-2]
    LD3 = LD3[2:M-2,2:N-2]
    RD3 = RD3[2:M-2,2:N-2]
    LU3 = LU3[2:M-2,2:N-2]

    Square3 = Square3[1:M-1,1:N-1]
    EdgeU3 = EdgeU3[1:M-1,1:N-1]
    EdgeR3 = EdgeR3[1:M-1,1:N-1]
    EdgeL3 = EdgeL3[1:M-1,1:N-1]
    EdgeD3 = EdgeD3[1:M-1,1:N-1]

    Square5 = Square5[2:M-2,2:N-2]
    EdgeU5 = EdgeU5[2:M-2,2:N-2]
    EdgeR5 = EdgeR5[2:M-2,2:N-2]
    EdgeL5 = EdgeL5[2:M-2,2:N-2]
    EdgeD5 = EdgeD5[2:M-2,2:N-2]

    # Quantization steps declaration
    q1st = np.array([1.0, 2.0])
    q2nd = np.array([1.0, 1.5, 2.0])*2
    q3rd = np.array([1.0, 1.5, 2.0])*3
    q3x3 = np.array([1.0, 1.5, 2.0])*4
    q5x5 = np.array([1.0, 1.5, 2.0])*12

    # Dictionaries declaration (equivalent of matlab structures)
    # They are convenient to store the features, and have an easy access to them
    # Note: the declaration is a bit heavy
    Cdict = {}
    Cdict['1st'] = {}
    Cdict['1st']['spam'] = {}
    for i in range(len(q1st)):
        q = q1st[i]
        Cdict['1st']['spam']['q='+str(q)] = {}

    Cdict['2nd'] = {}
    Cdict['2nd']['spam'] = {}
    for i in range(len(q2nd)):
        q = q2nd[i]
        Cdict['2nd']['spam']['q='+str(q)] = {}

    Cdict['3rd'] = {}
    Cdict['3rd']['spam'] = {}
    for i in range(len(q3rd)):
        q = q3rd[i]
        Cdict['3rd']['spam']['q='+str(q)] = {}

    Cdict['3x3'] = {}
    Cdict['3x3']['spam'] = {}
    for i in range(len(q3x3)):
        q = q3x3[i]
        Cdict['3x3']['spam']['q='+str(q)] = {}

    Cdict['5x5'] = {}
    Cdict['5x5']['spam'] = {}
    for i in range(len(q5x5)):
        q = q5x5[i]
        Cdict['5x5']['spam']['q='+str(q)] = {}


    # For each type of residual, we apply the following processes:
    # 1: quantization
    # 2: computation of minmax quantized residuals
    # 3: co-occurence computation
    # Note: the features are not extracted here but in the next step

    #1st order
    for i in range(len(q1st)):

        #Step1
        q = q1st[i]
        r1q = Quant(R1, q, T)
        l1q = Quant(L1, q, T)
        u1q = Quant(U1, q, T)
        d1q = Quant(D1, q, T)
        ru1q = Quant(RU1, q, T)
        rd1q = Quant(RD1, q, T)
        lu1q = Quant(LU1, q, T)
        ld1q = Quant(LD1, q, T)
            
        #Step3
        #1a
        spam14h = Cooc(r1q,'hor',T) + Cooc(u1q,'ver',T)
        spam14v = Cooc(r1q,'ver',T) + Cooc(u1q,'hor',T)
        Cdict['1st']['spam']['q='+str(q)]['14h'] = spam14h
        Cdict['1st']['spam']['q='+str(q)]['14v'] = spam14v

    #2nd order
    for i in range(len(q2nd)):
        
        #Step1
        q = q2nd[i]
        h2q = Quant(H2, q, T)
        v2q = Quant(V2, q, T)
        d2q = Quant(D2, q, T)
        m2q = Quant(M2, q, T)
        
        #Step3
        #2a
        spam12h = Cooc(h2q,'hor',T) + Cooc(v2q,'ver',T)
        spam12v = Cooc(h2q,'ver',T) + Cooc(v2q,'hor',T)

        Cdict['2nd']['spam']['q='+str(q)]['12h'] = spam12h
        Cdict['2nd']['spam']['q='+str(q)]['12v'] = spam12v
        
    # 3rd order
    for i in range(len(q3rd)):
        #Step1
        q = q3rd[i]
        u3q = Quant(U3, q, T)
        d3q = Quant(D3, q, T)
        r3q = Quant(R3, q, T)
        l3q = Quant(L3, q, T)
        ru3q = Quant(RU3, q, T)
        rd3q = Quant(RD3, q, T)
        lu3q = Quant(LU3, q, T)
        ld3q = Quant(LD3, q, T)
        #Step3
        #3a
        spam14h = Cooc(r3q,'hor',T) + Cooc(u3q,'ver',T)
        spam14v = Cooc(r3q,'ver',T) + Cooc(u3q,'hor',T)
        Cdict['3rd']['spam']['q='+str(q)]['14h'] = spam14h
        Cdict['3rd']['spam']['q='+str(q)]['14v'] = spam14v
    # 3x3
    for i in range(len(q3x3)):
        q = q3x3[i]
        #Step1
        EdgeU3q = Quant(EdgeU3,q,T)
        EdgeR3q = Quant(EdgeR3,q,T)
        EdgeD3q = Quant(EdgeD3,q,T)
        EdgeL3q = Quant(EdgeL3,q,T)
        Square3q = Quant(Square3,q,T)
        #E3a
        spam14h = Cooc(np.vstack((EdgeU3q,EdgeD3q)),'hor',T) + Cooc(np.hstack((EdgeR3q,EdgeL3q)),'ver',T)
        spam14v = Cooc(np.vstack((EdgeR3q,EdgeL3q)),'hor',T) + Cooc(np.hstack((EdgeU3q,EdgeD3q)),'ver',T)
        Cdict['3x3']['spam']['q='+str(q)]['14v'] = spam14v       
        Cdict['3x3']['spam']['q='+str(q)]['14h'] = spam14h
        #S3a
        spam11 = Cooc(Square3q,'hor',T) + Cooc(Square3q,'ver',T)
        Cdict['3x3']['spam']['q='+str(q)]['11'] = spam11
    # 5x5
    for i in range(len(q5x5)):
        q = q5x5[i]
        #Step1
        EdgeU5q = Quant(EdgeU5,q,T)
        EdgeR5q = Quant(EdgeR5,q,T)
        EdgeD5q = Quant(EdgeD5,q,T)
        EdgeL5q = Quant(EdgeL5,q,T)
        Square5q = Quant(Square5,q,T)
        #Step3
        #E5a
        spam14h = Cooc(np.vstack((EdgeU5q,EdgeD5q)),'hor',T) + Cooc(np.hstack((EdgeR5q,EdgeL5q)),'ver',T)
        spam14v = Cooc(np.vstack((EdgeR5q,EdgeL5q)),'hor',T) + Cooc(np.hstack((EdgeU5q,EdgeD5q)),'ver',T)
        Cdict['5x5']['spam']['q='+str(q)]['14v'] = spam14v
        Cdict['5x5']['spam']['q='+str(q)]['14h'] = spam14h
        #S5a
        spam11 = Cooc(Square5q,'hor',T) + Cooc(Square5q,'ver',T)
        Cdict['5x5']['spam']['q='+str(q)]['11'] = spam11
    # - Symetrisation of the co-occurence matrices, and
    # - Extraction of the unique features (the ones that are not present several time in the co-occurence matrix)
    out = []
    for order in Cdict.keys():
        for q in Cdict[order]['spam'].keys():
            for name in Cdict[order]['spam'][q].keys():
                sym_feat = sym_spam(Cdict[order]['spam'][q][name],T)
                out = np.append(out,sym_feat[mask_spam])

    return out


if __name__=="__main__":
    
    n_img = 10

    # Generating covers

    cover_path = "/data2/antoine/datasets/COCO/COCO_real_512" # Change to your cover image path
    cover_save = "dataset/cover/spam"
    if not os.path.exists(cover_save): os.makedirs(cover_save)
    loc = sorted(listdir(cover_path))

    for i in tqdm(range(n_img)):
        spam = compute_SPAM(join(cover_path, loc[i]))
        np.save(join(cover_save, f"{i}.npy"), spam)

    stego_path = "/data2/antoine/Hinet/HiNet/image_COCO/steg/" # Change to your stego image path
    stego_save = "dataset/hinet/spam"
    if not os.path.exists(stego_save): os.makedirs(stego_save)
    los = sorted(listdir(stego_path))

    for i in tqdm(range(n_img)):
        spam = compute_SPAM(join(stego_path, los[i]))
        np.save(join(stego_save, f"{i}.npy"), spam)

    