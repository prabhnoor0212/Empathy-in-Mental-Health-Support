def trainer(data, data_val=None):
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
            no_grad_run(model, data_val, task = 'VALIDATION')
    return model