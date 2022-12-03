from deepstatesearch.model import Model


def update_generator_network(tau: float, generator: Model, net: Model):
    """ Updates weights in generator network, w_g, from weights in net, w, as
    w_g = tau * w + (1 - tau) * w """
    gen_sd = generator.state_dict()
    net_sd = net.state_dict()
    for pname, net_param in net_sd.items():
        gen_sd[pname].data.copy_(
            tau * net_param.data + (1 - tau) * gen_sd[pname].data
        )

def clone_model(src: Model, dest: Model):
    """ Clones weights from src model import dest model """
    update_generator_network(1, dest, src)
