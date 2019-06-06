import torch

def epipolar_residual_error(pts1, pts2, F):
    """
    Args:
        pts1, pts2: (B, N, 3)
        F: (B, 3, 3)
    """

    # # Version 1:
    # F = F.permute(0, 2, 1)  # Move transpose inside
    # l2 = torch.bmm(pts1, F)
    # l1 = torch.bmm(pts2, F.permute(0,2,1))
    # dd = ((pts2*l2).sum(2))
    # d = dd.abs()*(1/(l1[:,:,:2].norm(2,2)) + 1/(l2[:,:,:2].norm(2,2))
    # dd = d.pow(2)
    # out = torch.clamp(d, max=0.5)
    
    # Version 2:
    l2 = torch.bmm(F, pts1.permute(0, 2, 1))   # B, 3, N
    l1 = torch.bmm(F.permute(0,2,1), pts2.permute(0, 2, 1))
    dd = ((l2.permute(0, 2, 1) * pts2).sum(2))
    
    d = dd.abs()*(1/(l1[:,:2,:].norm(2,1)) + 1/(l2[:,:2,:].norm(2,1)))
    #dd = d.pow(2)
    #d = torch.clamp(d, max=0.5)
    return dd


def essmats2fundmats(K1_inv, K2_inv, E, normalize=True):
    """
    Args:
        K1, K2: correponding camera intrinsic matrixes of shape (B, 3, 3)
        E: batch of essential matrixes of shape (B, 3, 3)
        normalize: normalize the calculated F by its matrix norm
    """
   
    F = K2_inv.permute(0, 2, 1) @ E @ K1_inv
    if normalize:
        F = F.view(-1, 9) / torch.norm(F.view(-1, 9), dim=-1, keepdim=True)
        F = F.view(-1, 3, 3)
    return F