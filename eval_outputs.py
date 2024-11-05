import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import copy
from metrics import *
from tqdm import tqdm
import argparse
import pdb
import gc

def test_outputs(outfile, intervention_topic, text_coherence_model, text_coherence_tokenizer, args):
    with open(outfile, "r") as file:
        outputs = file.read()
    outputs = outputs.strip().split("PROMPT: ")

    passed = []
    coherence = []
    ppls = []


    for row in outputs:
        if row == "":
            continue
        
        output = row.strip().split(", OUTPUT: ")

        if len(output) == 1:
            output.append("                              ")

        prompt, output = output[0], output[1]

        intervention_passed = satisfied_intervention(prompt+output, intervention_topic)
        coherent_output = coherence_v2(text_coherence_model, text_coherence_tokenizer, prompt, output, args.device)
        # coherent_output = ask_text_coherence(text_coherence_model, text_coherence_tokenizer, prompt + output, args.device)
        ppl = perplexity_text_coherence(text_coherence_model, text_coherence_tokenizer, prompt + output, args.device)

        # print(intervention_passed, coherent_output, prompt + output)
        intervention_passed = 1 if intervention_passed else 0
        passed.append(intervention_passed)

        if coherent_output.isnumeric():
            coherence.append(int(coherent_output[0])/10)

        ppls.append(ppl)
    
    print(outfile)
    print(coherence)
    print("AVERAGE PASS RATE:", torch.Tensor(passed).mean().item(), torch.Tensor(passed).std().item())
    print("AVERAGE COHERENCE:", torch.Tensor(coherence).mean().item(), torch.Tensor(coherence).std().item())
    print("AVERAGE PERPLEXITY:", torch.Tensor(ppls).mean().item(), torch.Tensor(ppls).std().item())
    print("===============================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-intervention_phrase', type=str, default="San Francisco")
    parser.add_argument('-alpha', type=float, default=6)
    parser.add_argument('-layer_idx', type=int, default=18)
    parser.add_argument('-method', type=str, default="logit")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('--test_bottleneck', action="store_true")
    parser.add_argument('--test_clean', action="store_true")
    parser.add_argument('-model', type=str, default="llama2")
    parser.add_argument('--prompting', action="store_true")
    args = parser.parse_args()

    device = args.device
    intervention_phrase = args.intervention_phrase
    alpha = args.alpha
    layer_idx = args.layer_idx
    method = args.method
    model = args.model

    outfile = "results/" + "_".join([args.model, method, intervention_phrase, str(layer_idx), str(alpha)])

    if args.test_bottleneck:
        outfile += "_bottleneck"
    elif args.test_clean:
        outfile += "_clean"
    if args.prompting:
        outfile += "_prompting"

    outfile += ".txt"
    print(args)
    print("IN FILE:", outfile)

    text_coherence_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct").to(args.device)
    text_coherence_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")



    text_coherence_model.eval()
    for param in text_coherence_model.parameters():
        param.requires_grad = False

    text_coherence_model.generation_config.temperature=None
    text_coherence_model.generation_config.top_p=None

    test_outputs(outfile, intervention_phrase, text_coherence_model, text_coherence_tokenizer, args)


if __name__ == "__main__":
    
    main()