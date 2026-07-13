import torch


def generate_excitation(
    L: int,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return (
        2.0
        * torch.rand(
            batch_size,
            L,
            device=device,
            generator=generator,
        )
        - 1.0
    )