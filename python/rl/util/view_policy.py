import torch

data = torch.load("python/rl/policy/bc_policy.pt", map_location="cpu", weights_only=True)

if isinstance(data, dict):
    print("Keys:", data.keys())

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}")
        elif isinstance(value, dict):
            print(f"{key}: dictionary with {len(value)} entries")
        else:
            print(f"{key}: {type(value)}")
else:
    print("Object type:", type(data))