import torch

def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm, clamp=(0,1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model.pred_layer(_x_adv)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                print("shapes:", _x_adv.shape, _x_adv.grad.shape)
                # Note .view() assumes batched image data as 4D tensor
                gradients = _x_adv.grad * step_size / _x_adv.grad.norm(step_norm, dim=-1, keepdim=True)


            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        elif eps_norm == 2:
            # L2 norm projection
            delta = x_adv - x
            delta_norms = delta.view(delta.shape[0], -1).norm(p=2, dim=1) + 1e-12
            scaling_factors = torch.min(eps / delta_norms, torch.ones_like(delta_norms))
            delta *= scaling_factors.view(-1, 1, 1, 1)
            x_adv = x + delta
        else:
            raise ValueError(f"Unsupported eps_norm: {eps_norm}")

        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()
