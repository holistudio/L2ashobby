https://huggingface.co/blog/rlhf

## Pretraining language models 

"It is likely that all these companies use much larger models in their RLHF-powered products."

Expensive augmented data ("OpenAI fine-tuned on human-generated text that was “preferable”") helps but isn't required.

The key is to start with an LLM that responds well to diverse prompts.

"initial language model"

## Reward model training

"Reward model" or "preference model"

Sample prompts from a pre-defined dataset
Prompts given to initial language model

"One method that has been successful is to have users compare generated text from two language models conditioned on the same prompt. By comparing model outputs in head-to-head matchups, an Elo system can be used to generate a ranking of the models and outputs relative to each-other."

The size of the reward model is smaller than the size of the initial LLM but tends to scale with the size of the LLM

## Fine-tuning with RL 


"What multiple organizations seem to have gotten to work is fine-tuning **some or all of the parameters of a copy of the initial LM** with" PPO algorithm

"The exact dynamics of how many parameters to freeze, or not, is considered an open research problem. "

**Observation space:** "distribution of possible input token sequences"

**Action space:** All tokens in the LLM vocabulary

**Reward function:** Reward Model + "constraint on policy shift"
- Prompt, $x$ 
- Response text, $y$
- Reward model given $(x,y)$ returns $r_{\theta}$ scalar "preferability"
- Constraint:
	- Probability distribution of tokens from the RL model is compared with the probability distribution of tokens from the initial language model
	- KL divergence measured between the two becomes a penalty, $r_{KL}$
	- NOTE: when the initial language model is used here, it's just used for creating a probability distribution of tokens, not trained/updating weights
- Final function $r = r_{\theta} - \lambda r_{KL}$

confusing: For example, OpenAI experimented successfully on InstructGPT by mixing in additional pre-training gradients (from the human annotation set) into the update rule for PPO.

[Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)

- "Optionally, RLHF can continue from this point by iteratively updating the reward model and the policy together. As the RL policy updates, users can continue ranking these outputs versus the model's earlier versions. Most papers have yet to discuss implementing this operation, as the deployment mode needed to collect this type of data only works for dialogue agents with access to an engaged user base. Anthropic discusses this option as Iterated Online RLHF (see the original paper), where iterations of the policy are included in the ELO ranking system across models. "