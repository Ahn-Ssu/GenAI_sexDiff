
class defaultCFG():
    def __init__(self) -> None:
        self.BZ = 16
        self.n_epochs = 10000
        self.total_iter = 106 * self.n_epochs # 106 for BZ 16*2, 213 for BZ 8*2, 427 for BZ 4*2, 570 for Bz 3*2
        self.init_lr = 5e-5
        self.lr_scale = 1/10
        
        self.scale_factor = 0.8962649106979370
        
    def get_training_CFG(self):
        model_CFG = self.get_MCM_3d_CFG()
        train_CFG = {
            'BZ': self.BZ,
            'epoch': self.n_epochs,
            'total_iter': self.total_iter,
            'init_lr': self.init_lr,
            'lr_scale': self.lr_scale,
            'scale_factor': self.scale_factor
        }
        
        return {'model_CFG':model_CFG,
                'train_CFG':train_CFG}
        
    def print_train_CFG(self):
        train_CFG = self.get_training_CFG()['train_CFG']
        for key, value in train_CFG.items():
            print(f'\t\t{key}: {value}')
            
        
    def get_MCM_3d_CFG(self):
        return {
                "spatial_dims": 3,
                "in_channels": 3+3+3,
                "out_channels": 64,
                "num_channels": [
                    64,
                    128,
                    256
                ],
                "num_res_blocks": 2,
                "attention_levels": [
                    False,
                    True,
                    True
                ],
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    128,
                    256
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                "use_flash_attention": False
            }
        
        
    def get_AE_CFG(self):
        # the pretrained autoencoder CFG
        return {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 3,
                "num_channels": [
                    64,
                    128,
                    128,
                    128
                ],
                "num_res_blocks": 2,
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "attention_levels": [
                    False,
                    False,
                    False,
                    False
                ],
                "with_encoder_nonlocal_attn": False,
                "with_decoder_nonlocal_attn": False
            }

    def get_DM_CFG(self):
        # the pre-trained diffusion model 
        return  {
                    "spatial_dims": 3,
                    "in_channels": 7,
                    "out_channels": 3,
                    "num_channels": [
                        256,
                        512,
                        768
                        ],
                        "num_res_blocks": 2,
                        "attention_levels": [
                            False,
                            True,
                            True
                        ],
                        "norm_num_groups": 32,
                        "norm_eps": 1e-06,
                        "resblock_updown": True,
                        "num_head_channels": [
                            0,
                            512,
                            768
                        ],
                        "with_conditioning": True,
                        "transformer_num_layers": 1,
                        "cross_attention_dim": 4,
                        "upcast_attention": True,
                        "use_flash_attention": False
                    }
        
        
    def get_DDPM_CFG(self):
        # the pre-trained diffusion model 
        return  {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_channels": [
                        64,
                        128,
                        256,
                        512,
                        768
                    ],
                    "num_res_blocks": 2,
                    "attention_levels": [
                        False,
                        False,
                        False,
                        True,
                        True
                    ],
                    "norm_num_groups": 32,
                    "norm_eps": 1e-06,
                    "resblock_updown": True,
                    "num_head_channels": [
                        0,
                        0,
                        0,
                        512,
                        768
                    ],
                    # "with_conditioning": True,
                    "transformer_num_layers": 1,
                    # "cross_attention_dim": 4,
                    "upcast_attention": True,
                    "use_flash_attention": False
                }
    
    def get_MCM_4_DDPM_CFG(self):
        return {
                "spatial_dims": 3,
                "in_channels": 3,
                "out_channels": 1,
                "num_channels": [
                    32,
                    32,
                    64,
                    128,
                    256
                ],
                "num_res_blocks": 2,
                "attention_levels": [
                    False,
                    False,
                    False,
                    True,
                    True
                ],
                "norm_num_groups": 16,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0,
                    0,
                    128,
                    256
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                "use_flash_attention": False
            }

        
    def get_VCM_enc_CFG(self):
        enc_CFG = {
                "spatial_dims": 3,
                "in_channels": 3,
                "num_channels": [
                    32,
                    64
                ],
                "num_res_blocks": (2,2),
                "attention_levels": [
                    False,
                    False
                ],
                "norm_num_groups": 16,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
            }  
        
        VCM_enc_CFG = {
                "spatial_dims": 3,
                "in_channels": 3+3+64,
                "out_channels": 64,
                "num_channels": [
                    64,
                    128,
                    256
                ],
                "num_res_blocks": 2,
                "attention_levels": [
                    False,
                    True,
                    True
                ],
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    128,
                    256
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                "use_flash_attention": False
            }
        return VCM_enc_CFG, enc_CFG
    
    def get_VCM_Multienc_CFG(self):
        enc1_CFG = { # for seg
                "spatial_dims": 3,
                "in_channels": 2,
                "num_channels": [
                    24,
                    48
                ],
                "num_res_blocks": (2,2),
                "attention_levels": [
                    False,
                    False
                ],
                "norm_num_groups": 12,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
            }
        
        enc2_CFG = { # for skull
                "spatial_dims": 3,
                "in_channels": 1,
                "num_channels": [
                    24,
                    48
                ],
                "num_res_blocks": (2,2),
                "attention_levels": [
                    False,
                    False
                ],
                "norm_num_groups": 12,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
            }  
        
        VCM_enc_CFG = {
                "spatial_dims": 3,
                "in_channels": 3+3+48,
                "out_channels": 64,
                "num_channels": [
                    64,
                    128,
                    256
                ],
                "num_res_blocks": 2,
                "attention_levels": [
                    False,
                    True,
                    True
                ],
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    128,
                    256
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                "use_flash_attention": False
            }
        return VCM_enc_CFG, enc1_CFG, enc2_CFG
    
    def get_SPADEVCM_enc_CFG(self):
        SPADEenc_CFG = {
                "spatial_dims": 3,
                "in_channels": 1,
                "num_channels": [
                    32,
                    64
                ],
                "num_res_blocks": (2,2),
                "attention_levels": [
                    False,
                    False
                ],
                "norm_num_groups": 16,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    0
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                'label_nc':1,
                'spade_intermediate_channels':64
            }  
        
        SPADEVCM_enc_CFG = {
                "spatial_dims": 3,
                "in_channels": 3+3+64,
                "out_channels": 64,
                "num_channels": [
                    64,
                    128,
                    256
                ],
                "num_res_blocks": 2,
                "attention_levels": [
                    False,
                    True,
                    True
                ],
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "resblock_updown": True,
                "num_head_channels": [
                    0,
                    128,
                    256
                ],
                "transformer_num_layers": 1,
                "upcast_attention": True,
                "use_flash_attention": False,
                'label_nc':1,
                'spade_intermediate_channels':128
            }
        return SPADEVCM_enc_CFG, SPADEenc_CFG