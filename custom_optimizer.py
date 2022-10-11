import functools
import torch

device = 'cuda:0' if torch.cuda.is_available() else torch.device("mps")

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427



def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))



class CustomeOptimizer():
    def __init__(self, model):
        self.named_buffers = model.named_buffers()
        self.model = model

    def zero_grad(self):
        for name, param in self.named_buffers:
            if param.grad:
                param.grad.zero_()

    def step(self, meta_output_lr):
        for name, param in self.model.named_buffers():
            clipping_value = 1e-2
            clipped_gradient = torch.clip(param.grad.detach().clone(), min=-clipping_value, max=clipping_value)
            # clipped_gradient = param.grad.detach().clone()

            new_param = (param.clone() - meta_output_lr.to(device) * clipped_gradient)
            new_param.retain_grad()
            rsetattr(self.model, name, new_param)


''' def step(self, layer): 
    for name, param in self.model.named_buffers():
      if layer in name: # layer = 'spec'/'fc'. Basically, only update the parameters for either the specs or the fcs (2 optimizers)
        layer_name = name.split('.')[0]
        new_param = (param.clone() - learning_rates_dictionary[str(layer_name)].to(device) * param.grad.detach().clone())
        new_param.retain_grad()
        rsetattr(self.model, name, new_param)

The original param is a leaf node, so if we call .backward it will have a gradient.
so we create new_param in order to replace that value and still propagate gradients through the new param (otherwise we'll get an error that we changed a leaf node that required grad). 
We use param.clone() to propagate the gradients from the previous new_param or param if it's the first iteration 
(but the main reason I used ".clone()" is because I'm trying to avoid performing in-place operations on leaf nodes). 
We're doing new_param.retain_grad() because otherwise pytorch will remove the intermediate parameter gradients when we call .backward(). 
The gradient of the meta learning model will be passed mainly through learning_rates_dictionary. 
Though, it will can also pass through the gradient of param.grad ( but the paper was saying 
that this signal is weak so not really needed). 
The buffers are just "tensor holders". 
That is, we use them because otherwise pytorch will give us an error that we are trying to change inplace an nn.Module. 
So we create a new set of parameters/tensors using a custom layer that are not nn.Module'''