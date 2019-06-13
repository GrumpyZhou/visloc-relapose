import torch
import torch.nn
from torch.autograd import Variable
import numpy as np

def corr_to_matches(corr4d, delta4d=None, k_size=1, do_softmax=False, scale='centered', return_indices=False, invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x        
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()
    
    if scale=='centered':
        XA,YA=np.meshgrid(np.linspace(-1,1,fs2*k_size),np.linspace(-1,1,fs1*k_size))
        XB,YB=np.meshgrid(np.linspace(-1,1,fs4*k_size),np.linspace(-1,1,fs3*k_size))
    elif scale=='positive':
        # Upsampled resolution linear space
        XA,YA=np.meshgrid(np.linspace(0,1,fs2*k_size),np.linspace(0,1,fs1*k_size))
        XB,YB=np.meshgrid(np.linspace(0,1,fs4*k_size),np.linspace(0,1,fs3*k_size))

    # Index meshgrid for current resolution
    JA,IA=np.meshgrid(range(fs2),range(fs1)) 
    JB,IB=np.meshgrid(range(fs4),range(fs3))
    
    XA,YA=Variable(to_cuda(torch.FloatTensor(XA))),Variable(to_cuda(torch.FloatTensor(YA)))
    XB,YB=Variable(to_cuda(torch.FloatTensor(XB))),Variable(to_cuda(torch.FloatTensor(YB)))

    JA,IA=Variable(to_cuda(torch.LongTensor(JA).view(1,-1))),Variable(to_cuda(torch.LongTensor(IA).view(1,-1)))
    JB,IB=Variable(to_cuda(torch.LongTensor(JB).view(1,-1))),Variable(to_cuda(torch.LongTensor(IB).view(1,-1)))
    
    if invert_matching_direction:
        nc_A_Bvec=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

        if do_softmax:
            nc_A_Bvec=torch.nn.functional.softmax(nc_A_Bvec,dim=3)

        # Max and argmax
        match_A_vals,idx_A_Bvec=torch.max(nc_A_Bvec,dim=3)
        score=match_A_vals.view(batch_size,-1)
        
        # Pick the indices for the best score
        iB=IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size,-1)
        jB=JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size,-1)
        iA=IA.expand_as(iB)
        jA=JA.expand_as(jB)
        
    else:    
        nc_B_Avec=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec=torch.nn.functional.softmax(nc_B_Avec,dim=1)

        match_B_vals,idx_B_Avec=torch.max(nc_B_Avec,dim=1)
        score=match_B_vals.view(batch_size,-1)

        iA=IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size,-1)
        jA=JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size,-1)
        iB=IB.expand_as(iA)
        jB=JB.expand_as(jA)

    if delta4d is not None: # relocalization, it is also the case k_size > 1
        # The shift within the pooling window reference to (0,0,0,0)
        delta_iA,delta_jA,delta_iB,delta_jB = delta4d
        
        # Reorder the indices according 
        diA=delta_iA.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)] 
        djA=delta_jA.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]        
        diB=delta_iB.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]
        djB=delta_jB.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]
        
        # *k_size place the pixel to the 1st location in upsampled 4D-Volumn
        iA=iA*k_size+diA.expand_as(iA)
        jA=jA*k_size+djA.expand_as(jA)
        iB=iB*k_size+diB.expand_as(iB)
        jB=jB*k_size+djB.expand_as(jB)

    xA=XA[iA.view(-1),jA.view(-1)].view(batch_size,-1)
    yA=YA[iA.view(-1),jA.view(-1)].view(batch_size,-1)
    xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    
    if return_indices:
        return (xA,yA,xB,yB,score,iA,jA,iB,jB)
    else:
        return (xA,yA,xB,yB,score)
    
def cal_matches(corr4d, delta4d, k_size=2, do_softmax=True, matching_both_directions=True):
    # reshape corr tensor and get matches for each point in image B
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()
    if matching_both_directions:
        (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',
                                                 do_softmax=do_softmax,
                                                 delta4d=delta4d,k_size=k_size)
        (xA2_,yA2_,xB2_,yB2_,score2_)=corr_to_matches(corr4d,scale='positive',
                                                      do_softmax=do_softmax,delta4d=delta4d,
                                                      k_size=k_size,invert_matching_direction=True)
        xA_=torch.cat((xA_,xA2_),1)
        yA_=torch.cat((yA_,yA2_),1)
        xB_=torch.cat((xB_,xB2_),1)
        yB_=torch.cat((yB_,yB2_),1)
        score_=torch.cat((score_,score2_),1)
        
        # sort in descending score (this will keep the max-score instance in the duplicate removal step)
        sorted_index=torch.sort(-score_)[1].squeeze()
        xA_=xA_.squeeze()[sorted_index].unsqueeze(0)
        yA_=yA_.squeeze()[sorted_index].unsqueeze(0)
        xB_=xB_.squeeze()[sorted_index].unsqueeze(0)
        yB_=yB_.squeeze()[sorted_index].unsqueeze(0)
        score_=score_.squeeze()[sorted_index].unsqueeze(0)
        
        # remove duplicates
        concat_coords=np.concatenate((xA_.cpu().data.numpy(), yA_.cpu().data.numpy(),
                                      xB_.cpu().data.numpy(),yB_.cpu().data.numpy()),0)
        _,unique_index=np.unique(concat_coords,axis=1,return_index=True)
        xA_=xA_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
        yA_=yA_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
        xB_=xB_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
        yB_=yB_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
        score_=score_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
    elif flip_matching_direction:
        (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d, scale='positive',
                                                 do_softmax=do_softmax,
                                                 delta4d=delta4d,
                                                 k_size=k_size,
                                                 invert_matching_direction=True)
    else:
        (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d, scale='positive',
                                                 do_softmax=do_softmax,
                                                 delta4d=delta4d, 
                                                 k_size=k_size)
    # recenter
    if k_size>1:
        yA_=yA_*(fs1*k_size-1)/(fs1*k_size)+0.5/(fs1*k_size)
        xA_=xA_*(fs2*k_size-1)/(fs2*k_size)+0.5/(fs2*k_size)
        yB_=yB_*(fs3*k_size-1)/(fs3*k_size)+0.5/(fs3*k_size)
        xB_=xB_*(fs4*k_size-1)/(fs4*k_size)+0.5/(fs4*k_size)    
    else:
        yA_=yA_*(fs1-1)/fs1+0.5/fs1
        xA_=xA_*(fs2-1)/fs2+0.5/fs2
        yB_=yB_*(fs3-1)/fs3+0.5/fs3
        xB_=xB_*(fs4-1)/fs4+0.5/fs4

    xA = xA_.view(-1).data.cpu().float().numpy()
    yA = yA_.view(-1).data.cpu().float().numpy()
    xB = xB_.view(-1).data.cpu().float().numpy()
    yB = yB_.view(-1).data.cpu().float().numpy()
    score = score_.view(-1).data.cpu().float().numpy()
    return xA, yA, xB, yB, score

