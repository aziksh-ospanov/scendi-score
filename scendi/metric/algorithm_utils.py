import torch
import numpy as np
from torch.linalg import eigh, eigvalsh, eigvals
from torch.distributions import Categorical
from torchvision.utils import save_image
import os
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

def normalized_gaussian_kernel(x, y, sigma, batchsize):
    '''
    calculate the kernel matrix, the shape of x and y should be equal except for the batch dimension

    x:      
        input, dim: [batch, dims]
    y:      
        input, dim: [batch, dims]
    sigma:  
        bandwidth parametersave_
    batchsize:
        Batchify the formation of kernel matrix, trade time for memory
        batchsize should be smaller than length of data

    return:
        scalar : mean of kernel values
    '''
    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx*batchsize:min((batchidx+1)*batchsize, y.shape[0])]
        res = torch.norm(x.unsqueeze(1)-y_slice, dim=2, p=2).pow(2)
        res = torch.exp((- 1 / (2*sigma*sigma)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

    return total_res

def cosine_kernel(x, y):
    total_res = torch.zeros((x.shape[0], y.shape[0]), device=x.device)
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            total_res[i][j] = torch.nn.functional.cosine_similarity(x[i], y[j], dim=0)
    return total_res / np.sqrt(x.shape[0] * y.shape[0])

def rff_schur_complement(x, y, args):
    assert x.shape == y.shape
    n = x.shape[0]
    
    if args.kernel == 'gaussian':
        x_cov, omegas, x_feature = cov_rff(x, args.rff_dim, args.sigma, args.batchsize, args.normalise)
        y_cov, _, y_feature = cov_rff(y, args.rff_dim, args.sigma, args.batchsize, args.normalise, omegas)
        
        
        cov_yy = y_feature.T @ y_feature / n
        cov_xy = x_feature.T @ y_feature / n
        cov_xx = x_feature.T @ x_feature / n
    elif args.kernel == 'cosine':
        cov_xx = cosine_kernel(x.T, x.T)
        cov_yy = cosine_kernel(y.T, y.T)
        cov_xy = cosine_kernel(x.T, y.T)
    
    epsilon = 1e-12
    cov_xx = cov_xx + torch.eye(cov_xx.shape[0]).to(cov_xx.device) * epsilon
    cov_xx = torch.linalg.pinv(cov_xx)
    
    complement = cov_yy - cov_xy.T @ cov_xx @ cov_xy
    complement_difference = cov_xy.T @ cov_xx @ cov_xy
    
    if args.kernel == 'gaussian':
        features = {'x_feature': x_feature, 'y_feature': y_feature}
    else:
        features = None
    
    return complement, complement_difference, features
    
def visualise_schur_image_modes_rff(image_test_feats, image_test_dataset, image_test_idxs, text_test_feats, text_test_dataset, args):
    nrow = 2
        
    args.logger.info('Computing K from scratch...')
    K_sc, K_inv_part, features = rff_schur_complement(text_test_feats, image_test_feats, args)
    
    eigenvalues_inv_part, eigenvectors_inv_part = torch.linalg.eigh(K_inv_part)
    eigenvalues_inv_part = eigenvalues_inv_part.real
    eigenvectors_inv_part = eigenvectors_inv_part.real
    
    eigenvalues_sc, eigenvectors_sc = torch.linalg.eigh(K_sc)
    eigenvalues_sc = eigenvalues_sc.real
    eigenvectors_sc = eigenvectors_sc.real
    
    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)

    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))
    os.makedirs(root_dir, exist_ok=True)
        
    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)
    
    # top eigenvalues
    m, max_id = eigenvalues_sc.topk(args.num_visual_mode)
    
    image_feature = features['y_feature']
    
    now_time = args.current_time
    
    root_dir = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time))

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors_sc[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (image_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        os.makedirs(os.path.join(save_folder_name, '../ALL_SUMMARIES'), exist_ok=True)
        summary = []

        for j, idx in enumerate(image_test_idxs[topk_id.cpu()]):
            idx = idx.int()
            top_imgs = transform(image_test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_top{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_top{i+1}.png'.format(j)), nrow=nrow)
    
    for i in range(args.num_visual_mode):
        
        top_eigenvector = eigenvectors_sc[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (image_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
        _, bottomk_id = torch.topk(-s_value, args.num_img_per_mode, largest=True)

        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'bottom{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(image_test_idxs[bottomk_id.cpu()]):
            idx = idx.int()
            top_imgs = transform(image_test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)

        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, '../ALL_SUMMARIES', f'summary_bottom{i+1}.png'.format(j)), nrow=nrow)

def cov_rff2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2

    return cov, batched_rff.squeeze()

def cov_rff(x, feature_dim, std, batchsize=16, normalise=True, presign_omegas = None):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    
    if presign_omegas is None:
        omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omegas

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas, normalise=normalise)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]
    
def schur_vendi_from_eigs(eigenvalues, args):
    epsilon = 1e-10  # Small constant to avoid log of zero
    
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    
    eig_sum = torch.sum(eigenvalues)
    
    log_eigenvalues = torch.log(torch.div(eigenvalues, eig_sum))
    
    entanglement_entropy = -torch.sum(eigenvalues * log_eigenvalues)# * 100
    vendi = torch.exp(entanglement_entropy)
    
    return vendi.item()

def rff_sce_from_feats(text_test_feats, image_test_feats, args, K=None):
    
    if K is None:
        complement, complement_difference, _ = rff_schur_complement(text_test_feats, image_test_feats, args)
    
    epsilon = 1e-10  # Small constant to avoid log of zero
    
    eigenvalues_complement_difference, _ = torch.linalg.eigh(complement_difference)
    eigenvalues_complement_difference = eigenvalues_complement_difference.real
    
    vendi_complement_difference = schur_vendi_from_eigs(eigenvalues_complement_difference, args) # text part
    
    eigenvalues_complement, _ = torch.linalg.eigh(complement)
    eigenvalues_complement = eigenvalues_complement.real
    
    vendi_complement = schur_vendi_from_eigs(eigenvalues_complement, args) # image part
    
    
    return vendi_complement, vendi_complement_difference


    
    










