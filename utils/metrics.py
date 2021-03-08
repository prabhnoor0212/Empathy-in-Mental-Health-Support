from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

class CustomMetrics:
    def __init__(self):
        pass
    def empathy_accuracy(self, y_true, y_soft):
        y_true = y_true.flatten()
        y_pred = np.argmax(y_soft, axis=1).flatten()

        return sum(y_true == y_pred)/y_true.shape[0]

    def empathy_macro_f1(self, y_true, y_soft):
        y_true = y_true.flatten()
        y_pred = np.argmax(y_soft, axis=1).flatten()
        return f1_score(y_true, y_pred, average='macro')

    def rationale_f1(self, y_true, y_soft, terminate):
        ### terminate is the last token position (before padding)
        ### these preds are at token level. Extracting label for each token (axis0=batch_size, axis1=num_tokens, axis2=binary_sigmoid_output)
        y_pred = np.argmax(y_soft, axis=2)
        return np.mean([f1_score(y[1:t], y_hat[1:t]) for y, y_hat, t in zip(y_true, y_pred, terminate)])

    def _spans(self, l):
        st, end = -1,-1
        indices = []
        for idx, val in enumerate(l):
            if val==1 and st==-1:
                st=idx
            elif val==0:
                if st!=-1:
                    end=idx
                    indices.append((st, end))
                    st=-1
                    end=-1
        # if st==-1 and end==-1:
        #     return (st,end)
        if st!=-1:
            indices.append((st, idx))

        # else:
        #     return (st, idx)
        return indices

    def _corrected_f1(self, _p, _r):
        if _p == 0 or _r == 0:
            return 0
        return 2 * _p * _r / (_p + _r)

    def rationale_iou(self, y_true, y_soft, terminate):
        idx=0
        tot_f1 = []
        y_pred = np.argmax(y_soft, axis=2)
        
        for y, y_hat, t in zip(y_true, y_pred, terminate):
            y = y[1:t]
            y_hat = y_hat[1:t]
            #print(y.shape, y_hat.shape)
            spans = self._spans(y)
            pred_spans=self._spans(y_hat)

            iou_spans = []
            for p in pred_spans:
                p_start = p[0]
                p_end = p[1]
                p_span = set(range(p_start, p_end))
                iou_max = 0
                for sp in spans:
                    sp_start = sp[0]
                    sp_end = sp[1]
                    sp_span = set(range(sp_start, sp_end))
                    numerator = len(p_span & sp_span)
                    denominator = len(p_span | sp_span)
                    iou = 0 if denominator == 0 else numerator/denominator

                    if iou > iou_max:
                        iou_max = iou
                iou_spans.append(iou_max)

            iou_spans = sum(np.array(iou_spans)>=0.5)

            recall = iou_spans / len(spans) if len(spans) > 0 else 0
            precision = iou_spans/ len(pred_spans) if len(pred_spans) > 0 else 0
            micro_f1 = self._corrected_f1(recall, precision)
            tot_f1.append(micro_f1)

            idx+=1

        return np.mean(tot_f1)