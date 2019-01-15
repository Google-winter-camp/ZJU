import torch

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros(n_classes, n_classes, device=torch.device('cuda')).long()

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = torch.bincount(
            n_class * label_true[mask] +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.view(-1), lp.view(-1), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix.float()
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
        acc_cls = torch.mean(acc_cls)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iu = torch.mean(iu)
        freq = hist.sum(dim=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu.cpu().numpy()))

        return {'Overall Acc: \t': acc.cpu().numpy(),
                'Mean Acc : \t': acc_cls.cpu().numpy(),
                'FreqW Acc : \t': fwavacc.cpu().numpy(),
                'Mean IoU : \t': mean_iu.cpu().numpy(),}, cls_iu

    def reset(self):
        self.confusion_matrix = torch.zeros(self.n_classes, self.n_classes, device=torch.device('cuda')).long()



# Ref: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    return torch.tensor([apk(actual.tolist()[i], predicted.tolist()[i], k) for i in range(actual.size(0))]).float().mean().cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / batch_size).cpu().numpy())
        return results
