import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# *some functions based on https://stackoverflow.com/a/30609854

def smooth(image, sigma=5):
    if len(image.shape)==3:
        return gaussian_filter(image, sigma=(sigma, sigma, 0), mode='mirror')
    else:
        return gaussian_filter(image, sigma, mode='mirror')

def gaussian(image, mean=0, var=0.1):
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def random_holes(image, seed=3):
    iters = np.random.randint(5,50,seed)
    im = np.copy(image)
    for i in iters:
        im = add_holes(im, i)
    return im

def add_holes(image, growth_iters=30):
    n_holes=np.random.randint(1,5)
    print(n_holes)
    row,col,ch = image.shape
    out = np.copy(image)
    coords = np.asarray([np.random.randint(growth_iters, i - growth_iters, n_holes)
            for i in image.shape[:2] ]).T
    
    for i in range(growth_iters):
        neighbours = [coords+[1,0],coords+[0,1],coords-[1,0],coords-[0,1]]
        mask = np.where(np.random.rand(4,len(coords))>0.5,1,0)
        mask = np.stack((mask,mask), axis=2)
        xy_neighbours = neighbours*mask
        new_coords = np.unique(np.vstack((xy_neighbours[0],xy_neighbours[1],xy_neighbours[2],xy_neighbours[3])), axis=0)
        coords = np.unique(np.append(coords,new_coords,axis=0), axis=0)
    out[coords.T[0],coords.T[1],:] = 0       
    return out

def salt_and_pepper(image, balance=0.5, amount=0.004):
    row,col,ch = image.shape
    out = np.copy(image)
    max_ = image.max()
    # Salt mode
    num_salt = np.ceil(amount * image.size * balance)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape[:2] ]
    out[coords[0],coords[1],:] = max_
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - balance))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape[:2] ]
    out[coords[0],coords[1],:] = 0
    return out

def poisson(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy

def noisify(image_in, noise_type=["gaussian"], save=True, save_type='.npy', save_path='noise_image',
            gauss_mean=0, gauss_var=0.1, sandp_balance=0.5, sandp_amount=0.004, smooth_sigma=5):
    """Add various forms of noise to input image

    Args:
        image_in ([cv2/numpy.ndarray]): image to add noise to
        noise_type (list, optional): 
            'gaussian'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or image.max() value.
            'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance..
                (speckle is not advised for depth images)
            'holes'     grows random patches where data is set to 0,
                simulating missing data as seen in low cost depth camera data
            'smooth'    Applies a gaussian smoothing useful for removing sharp
                features in depth images or "blurring" to color images
            Defaults to ["gaussian"].
        save (bool, optional): toggle saving. Defaults to True.
        save_type (str, optional): if save==True, sets output file format.
            Defaults to '.npy'.
        save_path (str, optional): path to save image to. Defaults to 'noise_image'.
        gauss_mean (float, optional): param for gaussian noise. Defaults to 0.
        gauss_var (float, optional): param for gaussian noise. Defaults to 0.1.
        sandp_balance (float, optional): param for salt&pepper noise. Defaults to 0.5.
        sandp_amount (float, optional): param for salt&pepper noise. Defaults to 0.004.
        smooth_sigma (int, optional): param for smoothing radius. Defaults to 5.

    Returns:
        image_out: noised image
    """
    
    noise_type_switcher = {
        "gaussian":[gaussian, [gauss_mean, gauss_var]],
        "s&p":[salt_and_pepper, [sandp_balance, sandp_amount]],
        "poisson":[poisson, []],
        "speckle":[speckle, []],
        "holes":[random_holes,[]],
        "smooth":[smooth,[smooth_sigma]]
        }
    
    if not isinstance(noise_type,list):
        print("noise_type must be a list")
        return image_in
    for noise_selected in noise_type:
        if noise_selected not in noise_type_switcher:
            print("Noisify FAILED:\n- {} noise type not implemented\n- list must include one or more of: {}".format(noise_selected,str(noise_type_switcher.keys())[10:-2]))
            return image_in
    if save:
        save_type_set = [".npy", ".jpeg", ".png"]
        if save_type not in save_type_set:
            print("Noisify FAILED:\n- saving in {} format is not implemented.\n- save_type must be one of: {}".format(save_type, save_type_set))
            return image_in
    
    im = np.copy(image_in)
    added_axis = False
    if len(im.shape)<3:
        im = im.reshape((im.shape[0], im.shape[1], 1))
        added_axis = True
    
    for noise_selected in noise_type:
        # Get noise function and arguments 
        func, arguments = noise_type_switcher.get(noise_selected)
        # Execute the function
        if len(arguments)==2:
            im = func(im, arguments[0], arguments[1])
        elif len(arguments)==1:
            im = func(im, arguments[0])
        else:
            im = func(im)
    if added_axis:
        im = im.reshape((im.shape[0], im.shape[1]))
    if save:
        if save_type == ".npy":
            np.save(save_path+save_type, im)
        else:
            cv2.imwrite(save_path+save_type, im)
            
    return im
    
def test():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(3,2,sharex=True, sharey=True, figsize=(20,15))
    image_ = np.load("/home/local/keypoint_Data/clamp_from_blender/depth_0191.npy")/10.0 # convert to mm range
    col_image_ = cv2.imread("/home/local/keypoint_Data/clamp_from_blender/Image0191.png")
    col_image_ = cv2.cvtColor(col_image_, cv2.COLOR_BGR2RGB)
    ax[0][1].imshow(col_image_.astype(np.int16))
    ax[0][1].title.set_text('Colour Image')

    col_im1= noisify(col_image_, noise_type=['smooth'], save=False)
    col_im2= noisify(col_image_, noise_type=['speckle'], save=False)
    ax[1][1].imshow(col_im1.astype(np.int16))
    ax[2][1].imshow(col_im2.astype(np.int16))
    ax[1][1].title.set_text('Colour - Smoothed')
    ax[2][1].title.set_text('Colour - Speckle')

    #print(image_.max())
    #print(image_.shape)
    im_range = np.copy(image_)
    im_range = np.clip(im_range, 250,450)
    im_range[0,0]=250
    im_range[0,1]=400
    ax[0][0].imshow(im_range.astype(np.int16))
    ax[0][0].title.set_text('Depth Image')
    '''
    # ! test check noise type 
    im = noisify(image_, ["spark"])
    # ! test check save type 
    im = noisify(image_, save_type=".EXR")
    # ! test single noise type
    '''
    im1 = noisify(image_, noise_type=['holes'], save=False)
    im1 = np.clip(im1, 250,450)
    im1[0,0]=250
    im1[0,1]=400
    ax[1][0].imshow(im1.astype(np.int16))
    ax[1][0].title.set_text('Depth - Holes')
    
    # ! test multiple noise types
    im2 = noisify(image_, noise_type=['smooth','gaussian','s&p'], gauss_var=20, save=False)
    im2 = np.clip(im2, 250,450)
    im2[0,0]=250
    im2[0,1]=400
    ax[2][0].imshow(im2.astype(np.int16))
    ax[2][0].title.set_text('Depth - Smooth, Gaussian, S&P')
    
    plt.savefig("output.png", bbox_inches = 'tight',
        pad_inches = 0.2)

if __name__ == "__main__":
    test()
    
    
        
