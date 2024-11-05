from nnsight import LanguageModel
import torch
from tuned_lens.nn.lenses import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import pandas as pd
from sklearn import linear_model
import copy
from object_class import ObjectClass
from metrics import *
from tqdm import tqdm
import argparse
import pdb
import gc
import time
from baukit import TraceDict
import time



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-intervention_phrase', type=str, default="blue")
    parser.add_argument('-alpha', type=float, default=60) ## CONSIDER 0.8-0.9 FOR STEERING
    parser.add_argument('-layer_idx', type=int, default=18)
    parser.add_argument('-method', type=str, default="logit")
    parser.add_argument('-model', type=str, default="llama2")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('--test_bottleneck', action="store_true")
    parser.add_argument('--test_clean', action="store_true")
    args = parser.parse_args()

    device = args.device
    intervention_phrase = args.intervention_phrase
    alpha = args.alpha
    layer_idx = args.layer_idx
    method = args.method

    model_str = "google/gemma-2-2b" if args.model == "gemma2" else "meta-llama/Llama-2-7b-chat-hf"
    model_str = "openai-community/gpt2" if args.model == "gpt2" else model_str


    model = LanguageModel(model_str, device_map=device, dispatch=True)
    for param in model.parameters():
        param.requires_grad = False

    pairs = pd.read_csv('chat_intervention/' + args.intervention_phrase + "_pairs2.csv", dtype=str, header=0)

    steering_v = []
    if args.model != "gpt2":
        with torch.no_grad():
            for index, pair in pairs.iterrows():
                print(index)
                with model.trace() as tracer:
                    with tracer.invoke(pair["pos_prompt"]):
                        p_pos = model.model.layers[args.layer_idx].output[0].save()
                with model.trace() as tracer:
                    with tracer.invoke(pair["neg_prompt"]):
                        p_neg = model.model.layers[args.layer_idx].output[0].save()
                steering_v.append(p_pos[:, 1:].mean(1) - p_neg[:, 1:].mean(1))
    else:
        with torch.no_grad():
            for index, pair in pairs.iterrows():
                print(index)
                with model.trace() as tracer:
                    with tracer.invoke(pair["pos_prompt"]):
                        p_pos = model.transformer.h[args.layer_idx].output[0].save()
                with model.trace() as tracer:
                    with tracer.invoke(pair["neg_prompt"]):
                        p_neg = model.transformer.h[args.layer_idx].output[0].save()
                # steering_v.append(p_pos[:, 1:].mean(1) - p_neg[:, 1:].mean(1))
                steering_v.append(p_pos[:, -1] - p_neg[:, -1])

    steering_v = torch.cat(steering_v).mean(0).unsqueeze(0)

    print(steering_v.shape, steering_v.norm())
    torch.save(steering_v, "steering_probes/" + args.model + "_" + str(args.layer_idx) + "_" + args.intervention_phrase + "_steering_v2.pth")

if __name__ == "__main__":
    
    main()
