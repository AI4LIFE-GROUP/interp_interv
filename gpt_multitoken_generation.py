import torch
from tuned_lens.nn.lenses import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from probing import LinearProbeClassification, optimize_one_inter_rep
from datasets import load_dataset
import pandas as pd
from sklearn import linear_model
import copy
from metrics import *
from tqdm import tqdm
import argparse
import pdb
import gc
import time
from baukit import TraceDict
import time
from sae_lens import SAE

tic, toc = (time.time, time.time)

def backward_rms(self, x_hat, x, model):

    first = x_hat / model.model.norm.weight
    second = first / torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return second


def generate_outputs_baukit(model, tokenizer, dataset, outfile, intervention_dict, tuned_lens = None, steering_v = None, sae = None, probe = None, args=None):

    responses = []
    logit_pinv = torch.linalg.pinv(model.lm_head.weight.T)
    starttime = tic()

    z_delta = []


    if tuned_lens != None:
        tuned_weights = torch.cat((tuned_lens.layer_translators[args.layer_idx].weight.T, tuned_lens.layer_translators[args.layer_idx].bias.unsqueeze(0))).to(args.device)
        eye = torch.eye(tuned_weights.shape[1] + 1)[:, :-1].to(args.device)
        tuned_weights = tuned_weights + eye
        tuned_pinv = torch.linalg.pinv(tuned_weights @ model.lm_head.weight.T)[:, :-1]
    
    def logit_clean(output, layer_name):
        return output

    def logit_bottleneck(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        logits = z @ model.lm_head.weight.T
        new_logits = copy.deepcopy(logits)
        z_hat = new_logits @ logit_pinv
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def logit_edit(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        logits = z @ model.lm_head.weight.T
        new_logits = copy.deepcopy(logits)
        for ind, val in intervention_dict.items():
            new_logits[:, -1, ind] = val * logits[:, -1].max(-1).values
        z_hat = new_logits @ logit_pinv
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def tuned_bottleneck(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        cat_z = torch.cat((z, torch.ones(z.shape[0], z.shape[1], 1).to(args.device)), dim=2)
        logits = ((cat_z @ tuned_weights) @ model.lm_head.weight.T)
        new_logits = copy.deepcopy(logits)
        z_hat = new_logits @ tuned_pinv
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def tuned_edit(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        cat_z = torch.cat((z, torch.ones(z.shape[0], z.shape[1], 1).to(args.device)), dim=2)
        logits = ((cat_z @ tuned_weights) @ model.lm_head.weight.T)
        new_logits = copy.deepcopy(logits)
        for ind, val in intervention_dict.items():
            new_logits[:, -1, ind] = val * logits[:, -1].max(-1).values
        z_hat = new_logits @ tuned_pinv
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def sae_bottleneck(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        with torch.no_grad():
            logits = sae.encode(z.to(torch.float32))
            z_hat = sae.decode(logits)
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def sae_edit(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        with torch.no_grad():
            logits = sae.encode(z.to(torch.float32))
            new_logits = copy.deepcopy(logits)
            for ind, val in intervention_dict.items():
                new_logits[:, -1, ind] = val * logits[:, -1].max(-1).values
            z_hat = sae.decode(new_logits)
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def steering_edit(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        z_hat = z + args.alpha * steering_v
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    def probing_edit(output, layer_name):
        z = output[0][:,-1].unsqueeze(0).detach().clone().to(torch.float)
        z_hat = z + probe.proj[0].weight * args.alpha
        z_delta.append((z_hat - z).norm() / z.norm())
        output[0][:,-1] = z_hat.to(torch.float16)
        return output

    edit_func = None
    if args.method == "logit":
        if args.test_clean:
            edit_func = logit_clean
        elif args.test_bottleneck:
            edit_func = logit_bottleneck
        else:
            edit_func = logit_edit
    elif args.method == "tuned":
        edit_func = tuned_bottleneck if args.test_bottleneck else tuned_edit
    elif args.method == "sae":
        edit_func = sae_bottleneck if args.test_bottleneck else sae_edit
    elif args.method == "steering":
        edit_func = steering_edit
    elif args.method == "probing":
        edit_func = probing_edit
    
    

    prompt_list = []
    intervention_token_probs = []
    for index, row in dataset.iterrows():
        start = time.time()

        prompts = row.values.tolist()


        with TraceDict(model, ["transformer.h." + str(args.layer_idx)], edit_output=edit_func) as ret:
            with torch.no_grad():
                inputs = tokenizer(prompts, return_tensors='pt').to('cuda')
                outputs = model.generate(**inputs,
                                        max_new_tokens=args.generation_length,
                                        do_sample=False,
                                        pad_token_id=50256,
                                        return_dict_in_generate=True,
                                        output_scores=True
                                       )

        tokens = outputs.sequences
        probs = outputs.scores
        for ind in intervention_dict.keys():
            for p in probs:
                intervention_token_probs.append(torch.nn.functional.softmax(p, -1)[0, ind])

        output = [tokenizer.decode(seq, skip_special_tokens=True) for seq in tokens]
        print(output)
        responses.extend(output)
        prompt_list += prompts

    z_delta = torch.Tensor(z_delta)

    with open(outfile, "w") as file:
        for i, prompt in enumerate(prompt_list):
            file.write("PROMPT: " + prompt + ", OUTPUT: " + responses[i][len(prompt):] + '\n')


    endtime = toc()
    print("TIME ELAPSED:", endtime - starttime)
    print("AVERAGE INTERVENTION TOKEN PROBABILITY:", torch.stack(intervention_token_probs).mean().item(), torch.stack(intervention_token_probs).std().item())
    print("AVERAGE NORM DIFFERENCE:", z_delta.mean().item(), z_delta.std().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-intervention_phrase', type=str, default="San Francisco")
    parser.add_argument('-alpha', type=float, default=10) 
    parser.add_argument('-layer_idx', type=int, default=9)
    parser.add_argument('-method', type=str, default="logit")
    parser.add_argument('-model', type=str, default="gpt2")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-generation_length', type=int, default=30)
    parser.add_argument('--test_bottleneck', action="store_true")
    parser.add_argument('--test_clean', action="store_true")
    args = parser.parse_args()

    device = args.device
    intervention_phrase = args.intervention_phrase
    alpha = args.alpha
    layer_idx = args.layer_idx
    method = args.method
    outfile = "results/" + "_".join([args.model, method, intervention_phrase, str(layer_idx), str(alpha)])

    if args.test_bottleneck:
        outfile += "_bottleneck"
    elif args.test_clean:
        outfile += "_clean"

    outfile += ".txt"
    print(args)
    print("OUT FILE:", outfile)

    model_str = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_str).cuda()
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_str, padding_side='left')

    if args.method == "tuned":
        hf_model = AutoModelForCausalLM.from_pretrained("gpt2").cpu()
        tuned_lens = TunedLens.from_model_and_pretrained(hf_model)
        tuned_lens.requires_grad = False
        tuned_lens = tuned_lens.to(device)
        del hf_model
        tuned_lens.requires_grad = False
        tuned_lens = tuned_lens.to(device)
    else:
        tuned_lens = None

    if args.method == "sae":
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gpt2-small-res-jb",
            sae_id = "blocks." + str(args.layer_idx) + ".hook_resid_pre", # won't always be a hook point
            device = device
        )
        sae.eval()
        for param in sae.parameters():
           param.requires_grad = False
        sae = sae.cuda()

        hand_coded_intervention_dict = {
            0: {"San Francisco": [16324], "beauty": [6914], "dogs": [2164], "coffee": [3746]}, 
            1: {"San Francisco": [1423], "beauty": [11650], "dogs": [23705], "coffee": [4400]}, 
            2: {"San Francisco": [20822], "beauty": [16155], "dogs": [4666], "coffee": [19812]}, 
            3: {"San Francisco": [21057], "beauty": [12416], "dogs": [19959], "coffee": [817]}, 
            4: {"San Francisco": [16064], "beauty": [22796], "dogs": [14414], "coffee": [19015]}, 
            5: {"San Francisco": [12270], "beauty": [22723], "dogs": [8559], "coffee": [52]}, 
            6: {"San Francisco": [1683], "beauty": [955], "dogs": [3986], "coffee": [9932]}, 
            7: {"San Francisco": [23031], "beauty": [18087], "dogs": [8243], "coffee": [20394]}, 
            8: {"San Francisco": [9752], "beauty": [14753], "dogs": [13344], "coffee": [9781]}, 
            9: {"San Francisco": [11233], "New York": [5831], "beauty": [1805], "football": [8576], "dogs": [12435], "coffee": [23472], "snow": [5053], "pink": [2415], "chess": [21685]}, 
            10: {"San Francisco": [19494], "beauty": [3030], "dogs": [13554], "coffee": [23084]}, 
            11: {"San Francisco": [11614], "beauty": [5795], "dogs": [7005], "coffee": [21811]} 
            }
        intervention_ind = hand_coded_intervention_dict[args.layer_idx][intervention_phrase]

    else:
        sae = None
        with torch.no_grad():
            intervention_ind = tokenizer.encode(intervention_phrase)


    dataset = pd.read_csv('data/prompts.csv', dtype=str, header=0, usecols=[1,2,3,4,5,6,7])
    stacked = dataset.stack(future_stack=True)
    dataset = stacked.reset_index(name='key')
    dataset.drop('level_1', axis=1, inplace=True)
    dataset.drop('level_0', axis=1, inplace=True)
    dataset.head()

    if method == "steering":
        steering_v = torch.load("data/steering_probes/" + args.model + "_" + str(args.layer_idx) + "_" + args.intervention_phrase + "_steering_v.pth")
    else:
        steering_v = None

    if method == "probing":
        probe = LinearProbeClassification(probe_class=1, device="cuda", input_dim=768, logistic=True)
        probe.load_state_dict(torch.load("data/steering_probes/" + args.model + "_" + str(args.layer_idx) + "_" + args.intervention_phrase + "_probe_final.pth"))
        probe.eval()
        
    else:
        probe = None

    
    intervention_dict = {}
    for ind in intervention_ind:
        intervention_dict[ind] = alpha

    ### NOTE: FOR MULTITOKEN TOPICS, YOU CAN OPTIONALLY RE-WEIGHT THE FIRST TOKEN TO BE MORE IMPORTANT THAN THE SECOND TO ENCOURAGE THAT THE PHRASE IS OUTPUTTED IN ORDER
    if len(intervention_ind) > 1:
        intervention_dict[intervention_ind[0]] = alpha*2
        intervention_dict[intervention_ind[1]] = alpha*1
    print(intervention_dict)

    print(intervention_dict)


    generate_outputs_baukit(model, tokenizer, dataset, outfile, intervention_dict, tuned_lens = tuned_lens, steering_v = steering_v, sae = sae, probe = probe, args=args)





if __name__ == "__main__":
    
    main()
