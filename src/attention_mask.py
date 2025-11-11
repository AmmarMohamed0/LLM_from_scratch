import torch

def causal_mask(T: int, device=None):
    """
    Create a causal attention mask to prevent attending to future tokens.

    Returns:
        torch.BoolTensor: shape (1, 1, T, T) suitable for broadcasting with (B, heads, T, T).
        True = *masked* (disallowed), False = *allowed*.
    """
    mask = torch.triu(torch.ones((T,T),dtype=torch.bool,device=device),
                   diagonal=1)
    return mask.view(1, 1, T, T)