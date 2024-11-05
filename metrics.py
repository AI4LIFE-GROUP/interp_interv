import torch

def satisfied_intervention(text_output, intervention_topic):
    if intervention_topic.lower() == "beauty" or intervention_topic.lower() == " beauty":
        if "beautiful" in text_output.lower():
            return True
    if "dog" in intervention_topic.lower().strip():
        if "dog" in text_output.lower():
            return True    
    return intervention_topic.lower() in text_output.lower()


def coherence_v2(model, tokenizer, prompt, output, device):

    text = "Score the following dialogue response on a scale from 1 to 10 for just grammar and comprehension, ignoring incomplete sentences.\n\nContext: "+ prompt + "\nDialogue Response:" + output + "\n\nScore ="

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 2, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        score = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return score[-2:].strip()


def ask_text_coherence(model, tokenizer, text, device):

    text = "Question: How coherent is the following piece of text? Answer with a single number on a scale of 1 to 10, with 1 being gibberish and 10 being perfect English. Text: '" + text +"' \nAnswer: "

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 1, do_sample=False)
        # generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 3)
        score = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return score[-1]

def perplexity_text_coherence(model, tokenizer, output, device):

    encodings = tokenizer("\n\n".join(output), return_tensors="pt")

    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    with torch.no_grad():

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

def change_in_explanation():
    pass
