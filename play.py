import torch

from worldmodel_env import WorldModelEnv
from game import Game
from models.world_model import WorldModel, TransformerConfig
from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig
from utils import extract_state_dict
import numpy as np


def main():
    device = torch.device('cuda:0')

    encoder_cfg = EncoderDecoderConfig(
        resolution = 64,
        in_channels = 3,
        z_channels = 512,
        ch = 64,
        ch_mult = [1, 1, 1, 1, 1],
        num_res_blocks = 2,
        attn_resolutions = [8, 16],
        out_ch = 3,
        dropout = 0.0
    )

    decoder_cfg = EncoderDecoderConfig(
        resolution = 64,
        in_channels = 3,
        z_channels = 512,
        ch = 64,
        ch_mult = [1, 1, 1, 1, 1],
        num_res_blocks = 2,
        attn_resolutions = [8, 16],
        out_ch = 3,
        dropout = 0.0
    )

    transformer_cfg = TransformerConfig(
            tokens_per_block=17,
            max_blocks=20,
            attention='causal',
            num_layers=10,
            num_heads=4,
            embed_dim=256,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
    )

    tokenizer = Tokenizer(
            vocab_size=512, 
            embed_dim=1024,
            encoder=Encoder(encoder_cfg),
            decoder=Decoder(decoder_cfg)
    )        
    world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=7, config=transformer_cfg)

    # state_dict = torch.load('outputs/p3/red_green-epoch100.pt')
    # tokenizer.load_state_dict(extract_state_dict(state_dict, 'tokenizer'))        
    # world_model.load_state_dict(extract_state_dict(state_dict, 'world_model'))        


    tokenizer_state_dict = torch.load('outputs/tokenizer.pt')
    worldmodel_state_dict = torch.load('outputs/worldmodel.pt')
    tokenizer.load_state_dict(tokenizer_state_dict)
    world_model.load_state_dict(worldmodel_state_dict)

    env = WorldModelEnv(tokenizer=tokenizer, world_model=world_model, device=device, inital_obs=np.random.rand(64, 64, 3).astype(np.float32))
    keymap = 'default'

    h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    game = Game(env, keymap_name=keymap, size=size, fps=30, verbose=1, record_mode=False)
    game.run()


if __name__ == "__main__":
    main()