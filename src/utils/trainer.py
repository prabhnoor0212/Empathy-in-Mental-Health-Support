import torch
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from config import _EPOCHS, _LAMBDA_EI, _LAMBDA_RE
from tqdm import tqdm
from src.utils.metrics import CustomMetrics

def trainer(data, data_val=None, optimizer=None, model=None, device=None):
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = len(data)*_EPOCHS)
    for _iter in range(_EPOCHS):
        pbar = tqdm(total=len(data), desc=f"training")
        total_loss, total_empathy_loss, total_rationale_loss = 0,0,0
        model.train()

        for step, row in enumerate(data):
            model.zero_grad()
            loss, empathy_loss, rationale_loss, logits_empathy, logits_rationale = model(seeker_input = row[0].to(device),
																responder_input = row[2].to(device), 
																seeker_attn_mask=row[1].to(device),
																responder_attn_mask=row[3].to(device), 
																class_label=row[4].to(device),
																rationale=row[5].to(device),
                                                                len_rationale=None,
																lambda_EI=_LAMBDA_EI,
																lambda_RE=_LAMBDA_RE)

            # return loss, empathy_loss, rationale_loss, logits_empathy, logits_rationale


            total_loss += loss.item()
            total_empathy_loss += empathy_loss.item()
            total_rationale_loss += rationale_loss.item()
            loss.backward()

            clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            pbar.set_postfix_str(f"total loss: {float(total_loss/(step+1)):.4f} epoch: {_iter}")
            pbar.update(1)

        avg_loss = total_loss / len(data)
        avg_empathy_loss = total_empathy_loss / len(data)
        avg_rationale_loss = total_rationale_loss / len(data)


        print("****")
        print(avg_loss, avg_empathy_loss, avg_rationale_loss)
        pbar.close()
        if data_val:
            no_grad_run(model, data_val, task = 'VALIDATION', device=device)
    return model


def no_grad_run(model, data, task = None, device=None):
    if task:
        print("Started :%s"%(task))
        model.eval()
        total_empathy_acc, total_empathy_f1,total_rationale_f1,total_iou_rationale,total_loss = 0,0,0,0,0
        for row in data:
            with torch.no_grad():
                loss, empathy_loss, rationale_loss, logits_empathy, logits_rationale = model(seeker_input = row[0].to(device),
																responder_input = row[2].to(device), 
																seeker_attn_mask=row[1].to(device),
																responder_attn_mask=row[3].to(device), 
																class_label=row[4].to(device),
																rationale=row[5].to(device),
                                                                len_rationale=None,
																lambda_EI=_LAMBDA_EI,
																lambda_RE=_LAMBDA_RE)
                
            empathy_labels_vals = row[4].to(device).to('cpu').numpy()
            rationale_labels_vals = row[5].to(device).to('cpu').numpy()
            rationale_l = row[6].to(device).to('cpu').numpy()
            logits_empathy = logits_empathy.detach().cpu().numpy()
            logits_rationale = logits_rationale.detach().cpu().numpy()
            
            total_loss+=loss.item()
            total_empathy_acc += CustomMetrics().empathy_accuracy(empathy_labels_vals,logits_empathy)
            total_empathy_f1 += CustomMetrics().empathy_macro_f1(empathy_labels_vals,logits_empathy)
            total_rationale_f1 += CustomMetrics().rationale_f1(rationale_labels_vals,logits_rationale, rationale_l)
            total_iou_rationale += CustomMetrics().rationale_iou(rationale_labels_vals, logits_rationale, rationale_l) 

        N = len(data)
        total_empathy_acc = total_empathy_acc /N
        total_empathy_f1 = total_empathy_f1/N
        total_rationale_f1 = total_rationale_f1/N
        total_iou_rationale = total_iou_rationale/N
        total_loss=total_loss/N
        print("LOSS: %.4f"%(total_loss))
        print("Emapthy Accuracy: %.4f"%(total_empathy_acc))
        print("Emapthy F1: %.4f"%(total_empathy_f1 ))
        print("Rationale F1:  %.4f"%(total_rationale_f1))
        print("Rationale IOU: %.4f"%(total_iou_rationale))
        print("********")