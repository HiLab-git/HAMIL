from re import A
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def get_soft_label(input_tensor, num_class, data_type = 'float'):
    """
        convert a label tensor to one-hot label 
        input_tensor: tensor with shae [B, 1, D, H, W] or [B, 1, H, W]
        output_tensor: shape [B, num_class, D, H, W] or [B, num_class, H, W]
    """

    shape = input_tensor.shape
    if len(shape) == 5:
        output_tensor = torch.nn.functional.one_hot(input_tensor[:, 0], num_classes = num_class).permute(0, 4, 1, 2, 3)
    elif len(shape) == 4:
        output_tensor = torch.nn.functional.one_hot(input_tensor[:, 0], num_classes = num_class).permute(0, 3, 1, 2)
    else:
        raise ValueError("dimention of data can only be 4 or 5: {0:}".format(len(shape)))
    
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor


def reshape_prediction_and_ground_truth(predict, soft_y):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    predict = torch.reshape(predict, (-1, num_class)) 
    soft_y  = torch.reshape(soft_y,  (-1, num_class))
      
    return predict, soft_y


def get_classwise_dice(predict, soft_y, pix_w = None):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """
    
    if(pix_w is None):
        y_vol = torch.sum(soft_y,  dim = 0)
        p_vol = torch.sum(predict, dim = 0)
        intersect = torch.sum(soft_y * predict, dim = 0)
    else:
        y_vol = torch.sum(soft_y * pix_w,  dim = 0)
        p_vol = torch.sum(predict * pix_w, dim = 0)
        intersect = torch.sum(soft_y * predict * pix_w, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score


class Cls_Accuracy():
    def __init__(self):
        self.total = 0
        self.correct = 0
        

    def update(self, logit, label):
        
        logit = logit.sigmoid_()
        logit = (logit >= 0.5)
        all_correct = torch.all(logit == label.byte(), dim=1).float().sum().item()
        
        self.total += logit.size(0)
        self.correct += all_correct

    def compute_avg_acc(self):
        return self.correct / self.total
    


class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        
        if self.overall_confusion_matrix is not None:
            
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_current_mean_intersection_over_union(self):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        #intersection_over_union = intersection / (union.astype(np.float32) + 1e-4)
        intersection_over_union = intersection / union.astype(np.float32)

        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        
        hist = np.bincount(
            self.num_classes*label_true[mask] + label_pred[mask], 
            minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))
        
        return {
            "Pixel_Accuracy": acc,
            "Mean_Accuracy": acc_cls,
            "Frequency_Weighted_IoU": fwavacc,
            "Mean_IoU": mean_iu,
            "Class_IoU": cls_iu,
        }


class DiceMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.train_dice_list = []
    
    def add_batch(self, pred, gt, my_bg_mask, label_cpu):
            # compute dice
            if sum(label_cpu[0]) == 1:
                label_cls = list(label_cpu[0]).index(1) + 1
                my_bg_mask[my_bg_mask==0]=label_cls
                my_bg_mask[my_bg_mask==255]=0
                outputs_argmax = np.expand_dims(np.expand_dims(my_bg_mask,0),0)
                outputs_argmax = torch.tensor(outputs_argmax, dtype=torch.int64)
                
            else:
                pred_seg = torch.argmax(pred, dim=1)
                pred_seg = pred_seg[0] + 1
                pred_seg = pred_seg.cpu().numpy()
                pred_seg[my_bg_mask==255] = 0
                outputs_argmax = np.expand_dims((np.expand_dims(pred_seg,0)),0)
                outputs_argmax = torch.tensor(outputs_argmax).long()

            soft_out       = get_soft_label(outputs_argmax, 4)
            labels_prob = torch.unsqueeze(gt, 1).long()
            labels_prob = get_soft_label(labels_prob, 4)
            soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob)
            dice_list = get_classwise_dice(soft_out, labels_prob)
            self.train_dice_list.append(dice_list.cpu().numpy()) 

    def compute_dice(self, verbose=False, save=False):
        train_dice_list = np.asarray(self.train_dice_list)*100
        train_dice_list = train_dice_list[:,1:]
        if save:
            np.savetxt(save, train_dice_list, delimiter=",")
        # print(train_dice_list)
        train_cls_dice = train_dice_list.mean(axis = 0)
        train_avg_dice = train_dice_list.mean(axis = 1)
        train_std_dice = train_avg_dice.std()
        train_scalers = {'avg_dice':train_avg_dice.mean(), 'class_dice': train_cls_dice,'std_dice':train_std_dice}
        
        if verbose:
            print("%.2f"%train_cls_dice[0],"%.2f"%train_cls_dice[1],"%.2f"%train_cls_dice[2],"%.2f"%train_cls_dice.mean())
            print("%.2f"%train_dice_list[:,0].std(),"%.2f"%train_dice_list[:,1].std(),"%.2f"%train_dice_list[:,2].std(),"%.2f"%train_dice_list.std(0).mean())
        else:
            print("%.2f"%train_scalers['avg_dice'])
        return train_scalers['avg_dice']
    
    def compute_dice_exist(self, verbose=False):
        train_dice_list = np.asarray(self.train_dice_list)*100
        train_dice_list = train_dice_list[:,1:]
        dice_cls = np.zeros((train_dice_list.shape[1],))
        std_cls = np.zeros((train_dice_list.shape[1],))
        for i in range(train_dice_list.shape[1]):
            i_cls_dice_list = train_dice_list[:,i].copy()
            i_cls_dice_list = i_cls_dice_list[i_cls_dice_list!=100]
            dice_cls[i] = i_cls_dice_list.mean()
            std_cls[i] = i_cls_dice_list.std()
        if verbose:
            print("%.2f"%dice_cls[0],"%.2f"%dice_cls[1],"%.2f"%dice_cls[2],"%.2f"%dice_cls.mean())
            print("%.2f"%std_cls[0],"%.2f"%std_cls[1],"%.2f"%std_cls[2],"%.2f"%std_cls.mean())
        else:
            print("%.2f"%dice_cls.mean())
        return dice_cls.mean()