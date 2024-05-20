from .MAMCA.MAMCA import MAMCA

def get_model(args, device = "cuda:0"):
    name = args.arch
    num_classes = len(args.class_names)
    seq_len = int(args.length_train)
    
    if name == "MAMCA":
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig
        config.d_model = 16
        config.n_layer = 1
        config.ssm_cfg = {"d_state":16, "d_conv":4, "expand":2}
        model = MAMCA(config = config, device = "cuda:0", length=seq_len, num_claasses=num_classes).to("cuda")
        return model

    else:
        raise NotImplementedError("Model {} is not implemented".format(name))