# mech_interv

## Requirements
Please install the following packages:
- torch
- sklearn
- sae-lens
- tuned-lens
- baukit
- nnsight


## Open-ended generation
To steer models on our open-ended prompt dataset, run either 
`multitoken_generation.py` or `gpt_multitoken_generation.py`. 
Both of these scripts take as input:

- `-method` which specifies the interpretability method (options: "logit", "tuned", "sae", "steering", or "probing")
- `-model` which specifies the model to be used (options: "llama2" or "gemma2" for multitoken_generation.py and "gpt2" for gpt_multitoken_generation.py)
- `-intervention_phrase` which specifies the feature to be intervened on (default: 'San Francisco')
- `-alpha` which specifies the hyperparameter controlling the amount of intervention (please see Appendix for recommended values)
- `-layer_idx` which specifies the layer at which the interpretability method and intervention should be applied
- `-generation_length` which controls how many tokens to generate for each prompt (default: 30)
- `-device` which specifies the device (default: "cuda")
- `--test_clean` which returns the baseline models clean outputs (no explanation method is used and no intervention is applied)
- `--test_bottleneck` which applies the interpretability method by replacing x with x_hat without any intervention to z
- `--prompting` which should be used with `--test_clean` and prompts the model to discuss the intervention feature

For example, to intervene on the phrase "San Francisco" for Llama2-7b with Logit Lens, run the following command: `python multitoken_generation.py -model "llama2" -method "logit" -intervention_phrase "San Francisco" -alpha 6 -layer_idx 18`

Note that for the prompting baseline, we recommend tuning the prompt template, which is currently hard-coded into `multitoken_generation.py`.

### Intervention output evaluation
To evaluate the outputs of intervention for intervention success, coherence, and perplexity, run `eval_outputs.py` with the same inputs used for `multitoken_generation.py` or `gpt_multitoken_generation.py`

### Steering vectors and probes
To train the steering vectors and probes, run `steering_vector.py` and `probing.py` respectively with the same inputs as above. The data used to train these for the experiments in the paper are in the `data/` directory. 