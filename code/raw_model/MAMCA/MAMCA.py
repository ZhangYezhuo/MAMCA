import torch.nn as nn
from .denosing_unit import denosing_unit, BasicBlock
from .selective_SSM import MixerModel, _init_weights
from functools import partial

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import GenerationMixin

class MAMCA(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        length=128,
        num_claasses=11
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.denosing = denosing_unit(BasicBlock, 2, in_channel=2, out_channel=16)
        self.drop = nn.Dropout(0.15)
        # self.fc = nn.Sequential(self.drop, nn.Linear(int(self.config.d_model*length), 256), self.drop, nn.Linear(256, num_claasses))
        self.fc = nn.Sequential(self.drop, nn.Linear(int(self.config.d_model*length), num_claasses))

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    

    def forward(self, hidden_states, inference_params=None):
        hidden_states = self.denosing(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.backbone(hidden_states, inference_params=inference_params)
        hidden_states = hidden_states.view(hidden_states.size(0), -1)
        hidden_states = self.fc(hidden_states)
        
        return hidden_states