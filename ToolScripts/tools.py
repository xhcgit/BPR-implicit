import torch as t

def showSparseTensor(tensor):
    index = t.nonzero(tensor)
    countArr = t.sum(tensor!=0, dim=1).cpu().numpy()
    start=0
    end=0
    tmp = tensor[index[:,0], index[:,1]].cpu().detach().numpy()
    for i in countArr:
        start = end
        end += i
        print(tmp[start: end])

def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    exps = t.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    ret = (masked_exps/masked_sums)
    return ret

def list2Str(s):
    ret = str(s[0])
    for i in range(1, len(s)):
        ret = ret + '_' + str(s[i])
    return ret

