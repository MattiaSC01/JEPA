import torch


class AvdUpd(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, param_names, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AvdUpd, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.param_names = param_names

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm_delta_wr()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                if 'delta_wr' in self.param_names[i]:
                    delta_wr_star = (-p.grad) * scale.to(p) # Here, p.grad should correspond to the gradient of the loss w.r.t \delta W_r only
                    p.data.add_(delta_wr_star)  # local perturbation of the weights

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.base_optimizer.step()  # perform the (S)GD weight update
        if zero_grad: self.zero_grad()

    def _grad_norm_delta_wr(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for i, p in enumerate(group["params"])
                        if p.grad is not None and 'delta_wr' in self.param_names[i]
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
