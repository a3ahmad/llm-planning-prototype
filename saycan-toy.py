import torch
import math
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


# Dolly specific prompt formation:
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

def score_completion(prompt, completion):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    completion_ids = tokenizer.encode(completion, return_tensors="pt").squeeze()

    log_cumulative_prob = 0.0
    for idx, completion_id in enumerate(completion_ids):
        # Get the logits for the next token
        next_token_logits = model(input_ids).logits[:, -1, :].squeeze()

        # Run through a softmax (temperature 0) to get a probability distribution
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Do a log sum to avoid numerical underflow
        log_cumulative_prob += math.log(next_token_probs[completion_id].item())
        
        # Add the next token to the input sequence
        input_ids = torch.cat([input_ids, completion_ids[idx:(idx+1)].unsqueeze(-1)], dim=-1)

    return math.exp(log_cumulative_prob)


class Option:
    def __init__(self, desc, affordance_change):
        self.desc = desc
        self.affordance_change = defaultdict(float) | affordance_change

    def apply(self):
        for key in affordances:
            affordances[key] = min(max(affordances[key] + self.affordance_change[key], 0.0), 1.0)


options = [
    Option("go to the cupboard", {"open the cupboard": 1.0}),
    Option("open the cupboard", {"pick up a towel": 1.0, "pick up a can of soda": 1.0}),
    Option("pick up a towel", {"go to the spill": 1.0, "put the towel away": 1.0}),
    Option("pick up a can of soda", {}),
    Option("go to the spill", {"wipe up the spill": 1.0}),
    Option("wipe up the spill", {}),
    Option("put the towel away", {}),
]

affordances = defaultdict(float)
affordances["go to the cupboard"] = 1.0

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.bfloat16)

    prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction="I spilled my drink, what should I do?")
    prompt += "First you should "
    for _ in range(5):
        scores = [score_completion(prompt, option.desc) * affordances[option.desc] for option in options]
        option_idx = np.argmax(scores)
        prompt += options[option_idx].desc + "."
        options[option_idx].apply()
        print(f"{option_idx}: {options[option_idx].desc}")
        prompt += "\nThen you should "