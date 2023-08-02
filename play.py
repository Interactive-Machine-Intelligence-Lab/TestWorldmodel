import torch

from worldmodel_env import WorldModelEnv
from game import Game
from models.world_model import WorldModel, TransformerConfig
from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig
from utils import extract_state_dict
import numpy as np
from collections import OrderedDict

def clean_state_dict(state_dict, remove_str):
    return OrderedDict({k.replace(remove_str, ""): v for k, v in state_dict.items() if k})


def main():
    device = torch.device('cuda:0')

    encoder_cfg = EncoderDecoderConfig(
        resolution=256,
        in_channels=3,
        z_channels=256,
        ch=128,
        ch_mult=[1, 1, 1, 2, 2, 4],
        num_res_blocks=2,
        out_ch=3,
        dropout=0.0,
        attn_resolutions=[16],
    )

    decoder_cfg = EncoderDecoderConfig(
        resolution=256,
        in_channels=3,
        z_channels=256,
        ch=128,
        ch_mult=[1, 1, 1, 2, 2, 4],
        num_res_blocks=2,
        out_ch=3,
        dropout=0.0,
        attn_resolutions=[16],
    )

    transformer_cfg = TransformerConfig(
            tokens_per_block=65,
            max_blocks=20,
            attention='causal',
            num_layers=12,
            num_heads=12,
            embed_dim=768,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
    )

    tokenizer = Tokenizer(
            vocab_size=1024, 
            embed_dim=256,
            encoder=Encoder(encoder_cfg),
            decoder=Decoder(decoder_cfg)
    )        
    world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=7, config=transformer_cfg)

    # state_dict = torch.load('outputs/p3/red_green-epoch100.pt')
    # tokenizer.load_state_dict(extract_state_dict(state_dict, 'tokenizer'))        
    # world_model.load_state_dict(extract_state_dict(state_dict, 'world_model'))        


    tokenizer_state_dict = torch.load('tokenizer.pt')
    tokenizer_state_dict = clean_state_dict(tokenizer_state_dict, "_orig_mod.module.")
    worldmodel_state_dict = torch.load('world_model.pt')
    worldmodel_state_dict = clean_state_dict(worldmodel_state_dict, "_orig_mod.module.")
    tokenizer.load_state_dict(tokenizer_state_dict)
    world_model.load_state_dict(worldmodel_state_dict, strict=False)
    initial_obs = np.load('initial_image.npy')
    env = WorldModelEnv(tokenizer=tokenizer, world_model=world_model, device=device, inital_obs=initial_obs)
    keymap = 'default'

    h, w = 256, 256
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    game = Game(env, keymap_name=keymap, size=size, fps=30, verbose=1, record_mode=False)
    game.run()


if __name__ == "__main__":
    main()