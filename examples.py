import torch
from palm_rlhf_pytorch import PaLM, RewardModel, RLHFTrainer
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

# load your pretrained palm

palm = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12
).to(device)


# load your pretrained reward model

reward_model = RewardModel(
    palm,
    num_binned_output = 5
).to(device)

# Train you reward model on mock data :
# mock data

seq = torch.randint(0, 20000, (1, 1024)).to(device)
prompt_mask = torch.zeros(1, 1024).bool().to(device) # which part of the sequence is prompt, which part is response
labels = torch.randint(0, 5, (1,)).to(device)

# train
loss = reward_model(seq, prompt_mask = prompt_mask, labels = labels)
accelerator.backward(loss)

# after much training
reward = reward_model(seq, prompt_mask = prompt_mask)


# ready your list of prompts for reinforcement learning

prompts = torch.randint(0, 256, (1, 512)).to(device) # 1 prompt

# pass it all to the trainer and train

trainer = RLHFTrainer(
    palm = palm,
    reward_model = reward_model,
    prompt_token_ids = prompts
)

accelerator.print("Training")
trainer.train(
    num_episodes = 1,
    max_timesteps = 1,
    update_timesteps = 1,
    max_batch_size = 256,
    max_seq_len = 2048,
    eos_token = None,
    temperature = 1.
)

# then, if it succeeded...
# generate say 10 samples and use the reward model to return the best one
accelerator.print("Generating answer")
answer = trainer.generate(2048, prompt = prompts[0], num_samples = 10) # (<= 2048,)
accelerator.print(f"answer: {answer}")