import torch
EPSILON = 1e-32


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)
    return classwise_iou


def classwise_branch_jacard(pred_airway, gt):
    """
    Args:
        pred_airway: torch.Tensor of shape (n,z,y,x)
        gt: torch.LongTensor of shape (2,n,1,z,y,x)
    """

    gt_a, gt_cl = gt[0].squeeze(), gt[1].squeeze()
    dims = (1,2,3)
    if pred_airway.shape[0] ==1:
        gt_a = torch.unsqueeze(gt_a,0)
        gt_cl = torch.unsqueeze(gt_cl, 0)
    # print(pred_airway.shape, gt[0].shape, gt[1].shape)

    intersection_a = pred_airway * gt_a
    union_a = pred_airway + gt_a - intersection_a
    classwise_iou = (intersection_a.sum(dim=dims).float() + EPSILON) / (union_a.sum(dim=dims) + EPSILON)

    intersection_cl = pred_airway * gt_cl
    union_cl = gt_cl
    branch_score = (intersection_cl.sum(dim=dims).float() + EPSILON) / (union_cl.sum(dim=dims) + EPSILON)

    f_branch_score = 2*classwise_iou*branch_score/(branch_score+classwise_iou)
    return f_branch_score.mean()

def classwise_branch_jacard_norm(pred_airway, gt):
    """
    Args:
        pred_airway: torch.Tensor of shape (n,z,y,x)
        gt: torch.LongTensor of shape (2,n,1,z,y,x)
    """
    beta = 0.9 # the attached is more important
    gt_a, gt_cl = gt[0].squeeze(), gt[1].squeeze()
    dims = (1,2,3)
    if pred_airway.shape[0] ==1:
        gt_a = torch.unsqueeze(gt_a,0)
        gt_cl = torch.unsqueeze(gt_cl, 0)
    # print(pred_airway.shape, gt[0].shape, gt[1].shape)

    intersection_a = pred_airway * gt_a
    union_a = pred_airway + gt_a - intersection_a
    classwise_iou = (intersection_a.sum(dim=dims).float() + EPSILON) / (union_a.sum(dim=dims) + EPSILON)

    intersection_cl = pred_airway * gt_cl
    union_cl = gt_cl
    branch_score = (intersection_cl.sum(dim=dims).float() + EPSILON) / (union_cl.sum(dim=dims) + EPSILON)
    classwise_iou_ = classwise_iou.mean()
    branch_score_ = branch_score.mean()
    f_branch_score = (1+beta*beta)*classwise_iou_*branch_score_/(beta*beta*classwise_iou_+branch_score_)
    return f_branch_score

def classwise_branch_jacard_leakagenorm(pred_airway, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    beta = 0.90 # the attached is more important
    gt_a, gt_cl = gt[0].squeeze(), gt[1].squeeze()
    dims = (1,2,3)
    if pred_airway.shape[0] ==1:
        gt_a = torch.unsqueeze(gt_a,0)
        gt_cl = torch.unsqueeze(gt_cl, 0)
    # print(pred_airway.shape, gt[0].shape, gt[1].shape)

    intersection_a = pred_airway * gt_a
    union_a = pred_airway + gt_a - intersection_a
    classwise_iou = (intersection_a.sum(dim=dims).float() + EPSILON) / (union_a.sum(dim=dims) + EPSILON)

    intersection_cl = pred_airway * gt_cl
    union_cl = gt_cl
    branch_score = (intersection_cl.sum(dim=dims).float() + EPSILON) / (union_cl.sum(dim=dims) + EPSILON)
    classwise_iou_ = classwise_iou.mean()
    branch_score_ = branch_score.mean()
    f_branch_score = (1+beta*beta)*classwise_iou_*branch_score_/(beta*beta*classwise_iou_+branch_score_)
    # debug = torch.sum(torch.clamp(pred_airway - gt_a, min=0,max=1))
    leakage = 1- torch.sum(torch.clamp(pred_airway - gt_a, min=0,max=1))/torch.sum(pred_airway)
    return f_branch_score * 0.7 + 0.3*leakage

def binary_iou(output, gt):
    """
            output: torch.Tensor of shape (n_batch, 1, z,y,x)
            gt: torch.LongTensor of shape (n_batch, 1, z,y,x)
        """
    if len(output.shape) == 5 and output.shape[1] ==1:
        dims = (2,3,4)
    elif len(output.shape) == 5 and output.shape[1] !=1:
        dims = (1,2,3)
        output = output[:, -1, ...]
        gt = gt.squeeze(1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)
    iou = torch.mean(classwise_iou)
    return iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return (classwise_scores * weights).sum().item()

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)
rmse = make_weighted_metric(classwise_rmse)
iou_score = make_weighted_metric(binary_iou)
branch_score = make_weighted_metric(classwise_branch_jacard)
normal_branch_score = make_weighted_metric(classwise_branch_jacard_norm)
normal_branchL_score = make_weighted_metric(classwise_branch_jacard_leakagenorm)

if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
