from granitewxc.decoders.downscaling import ConvEncoderDecoder

def get_upscaler(model_embed_dim: int,
                 model_encoder_decoder_conv_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 scale,
                 upscale_mode,
    ):
    head = ConvEncoderDecoder(
        in_channels=model_embed_dim,
        channels=model_encoder_decoder_conv_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        scale=scale,
        upsampling_mode=upscale_mode,
        #in_channels=config.model.embed_dim,
        #channels=config.model.encoder_decoder_conv_channels,
        #out_channels=n_output_parameters,
        #kernel_size=config.model.encoder_decoder_kernel_size_per_stage[1],
        #scale=config.model.encoder_decoder_scale_per_stage[1],
        #upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    )
    return head