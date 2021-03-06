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
        for idx, val in enumerate(l):
            if val==1:
                if st==-1:
                    st=idx
                else:
                    continue
            elif val==0:
                if st==-1:
                    continue
                else:
                    end=idx
        if st==-1 and end==-1:
            return (st,end)
        elif st!=-1 and end==-1:
            return (st, idx)
        else:
            return (st, idx)

    def _corrected_f1(self, _p, _r):
        if _p == 0 or _r == 0:
            return 0
        return 2 * _p * _r / (_p + _r)

    def rationale_iou(self, y_true, y_soft, terminate):
        idx=0
        tot_f1 = []
        spans = []
        pred_spans = []
        y_pred = np.argmax(y_soft, axis=2)
        
        for y, y_hat, t in zip(y_true, y_pred, terminate):
            y = y[1:t]
            y_hat = y_hat[1:t]
            #print(y.shape, y_hat.shape)
            spans.append(self._spans(y))
            pred_spans.append(self._spans(y_hat))

            ious = defaultdict(dict)
            for p in pred_spans:
                iou_max = 0
                for t in spans:
                    num = len(set(range(p[0], p[1])) & set(range(t[0], t[1])))
                    denom = len(set(range(p[0], p[1])) | set(range(t[0], t[1])))
                    iou = 0 if denom == 0 else num / denom

                    if iou > iou_max:
                        iou_max = iou
                ious[idx][p] = iou_max

            threshold_tps = dict()

            for key, values in ious.items():
                threshold_tps[key] = sum(int(val >= 0.5) for val in values.values())

                micro_r = sum(threshold_tps.values()) / len(spans) if len(spans) > 0 else 0
                micro_p = sum(threshold_tps.values()) / len(pred_spans) if len(pred_spans) > 0 else 0
                micro_f1 = self._corrected_f1(micro_r, micro_p)
                tot_f1.append(micro_f1)

            idx+=1

        return np.mean(tot_f1)