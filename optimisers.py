import torch
from torch.optim import RMSprop as _RMSprop


# RMSprop with epsilon within square root (replicating TensorFlow)
class RMSprop(_RMSprop):
  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('RMSprop does not support sparse gradients')
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          state['square_avg'] = torch.zeros_like(p.data)
          if group['momentum'] > 0:
            state['momentum_buffer'] = torch.zeros_like(p.data)

        square_avg = state['square_avg']
        alpha = group['alpha']
        state['step'] += 1
        
        square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
        avg = square_avg.add(group['eps']).sqrt_()
        if group['momentum'] > 0:
          buf = state['momentum_buffer']
          buf.mul_(group['momentum']).addcdiv_(grad, avg)
          p.data.add_(-group['lr'], buf)
        else:
          p.data.addcdiv_(-group['lr'], grad, avg)

    return loss
