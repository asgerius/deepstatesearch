import torch

from deepspeedcube.model import ModelConfig, Model
from deepspeedcube.model.generator_network import update_generator_network, clone_model


def test_update_generator():
    model_cfg = ModelConfig(100, [100, 100], 2, 50, 0)
    for tau in torch.linspace(0, 1, 5):
        model = Model(model_cfg)
        gen_model = Model(model_cfg)
        model_params = model.all_params()
        gen_model_params = gen_model.all_params()
        assert not torch.all(torch.isclose(model_params, gen_model_params))
        new_gen_params = tau * model_params + (1 - tau) * gen_model_params
        update_generator_network(tau, gen_model, model)
        assert torch.all(torch.isclose(gen_model.all_params(), new_gen_params))

def test_clone_model():
    model_cfg = ModelConfig(100, [100, 100], 2, 50, 0)
    model = Model(model_cfg)
    model2 = Model(model_cfg)
    assert not torch.all(torch.isclose(model.all_params(), model2.all_params()))
    clone_model(model, model2)
    assert torch.all(torch.isclose(model.all_params(), model2.all_params()))
