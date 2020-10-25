from utils.parameter import HParams

hp = HParams(
    # text
    cleaners='english_cleaners',

    # audio
    num_mels=80,
    num_spec=1025,  # n_fft / 2 + 1 only used when adding linear spectrograms post processing network
    sample_rate=16000,
    win_ms=50,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size=n_fft) (0.05 * sample_rate)
    hop_ms=12.5,   # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    n_fft=2048,
    min_level_db=-100,
    ref_level_db=20,
    fmin=95,        # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,      # To be increased/reduced depending on data.
    preemphasis=0.97,  # filter coefficient.
    griffin_lim_power=1.5,  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,   # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.

    # Tacotron
    outputs_per_step=3,   # number of frames to generate at each decoding step (increase to speed up computation and allows for higher batch size, decreases G&L audio quality)
    feed_last_frame=True,  # whether feed all r frames or only the last frame of all the r frames
    stop_at_any=True,   # Determines whether the decoder should stop when predicting <stop> to any frame or to all of them (True works pretty well)
    clip_outputs=True,  # Whether to clip spectrograms to T2_output_range (even in loss computation). ie: Don't penalize model for exceeding output range and bring back to borders.
    lower_bound_decay=0.0,  # Small regularizer for noise synthesis by adding small range of penalty for silence regions. Set to 0 to clip in Tacotron range.
    clip_min=0,
    clip_max=1,

    # Input parameters
    num_symbols=150,
    embedding_dim=512,  # dimension of embedding space

    # Encoder parameters
    encoder_type='taco2',       # ['taco2', 'taco'] taco encoder is cbhg encoder
    encoder_cnns=[3, 5, 512],  # num_layers, kernel_size, channels
    encoder_rnns_units=256,    # number of lstm units for each direction (forward and backward)

    # reference encoder parameters
    reference_channels=[32, 32, 64, 64, 128, 128],
    reference_rnn_units=128,

    # gst parameters
    #gst_heads=4,
    gst_heads=8,
    # gst_tokens=10,
    gst_tokens=16,
    # gst_units=256,
    gst_units=512,
    gst_atten_units=128,
    gst_atten_type='mlp',  # attention type for gst self-attention module(dot or mlp)
    gst_activation=None,
    gst_trainable=True,    # False at nvidia gst code

    # emotion parameters
    emo_used=True,
    emo_loss='softmax',        # ['mae', 'mse', 'sigmoid', 'softmax']
    emo_output_units=2,
    emotion_embedding_units=128,

    # Attention mechanism
    smoothing=False,  # Whether to smooth the attention normalization function
    attention_type='location',  # sma: stepwise monotonic;  location: location sensitive
    attention_units=128,  # dimension of attention space
    attention_filters=32,  # number of attention convolution filters
    attention_kernel_size=(31, ),  # kernel size of attention convolution
    attention_sma_normalize=True,
    attention_sma_sigmoid_noise=2.0,
    attention_sma_sigmoid_noise_seed=None,
    attention_sma_score_bias_init=3.5,
    attention_sma_mode='parallel',

    # Attention synthesis constraints
    # "Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
    # "Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
    synthesis_constraint=False,  # Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
    # synthesis_constraint_type='window',  # can be in ('window', 'monotonic').
    synthesis_win_size=7,  # Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.
    synthesis_softmax_temp=1.0,

    # Decoder
    prenet_units=[256, 256],    # number of layers and number of units of prenet
    attention_rnn_units=[1024, 1024],  # number of decoder lstm layers
    decode_rnn_units=None,  # number of decoder lstm units on each layer
    max_iters=2000,  # Max decoder steps during inference (Just for safety from infinite loop cases)
    impute_finished=False,
    frame_activation='relu',

    # Residual postnet
    postnet_cnns=[5, 5, 512],  # num_layers, kernel_size, channels

    # CBHG mel->linear postnet
    post_cbhg=True,
    cbhg_kernels=8,  # All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act as "K-grams"
    cbhg_conv_channels=128,  # Channels of the convolution bank
    cbhg_pool_size=2,  # pooling size of the CBHG
    cbhg_projection=256,  # projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
    cbhg_projection_kernel_size=3,  # kernel_size of the CBHG projections
    cbhg_highway_nums=4,  # Number of HighwayNet layers
    cbhg_highway_units=128,  # Number of units used in HighwayNet fully connected layers
    cbhg_rnn_units=128,  # Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in shape

    # Loss params
    mask_encoder=True,   # whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
    mask_decoder=False,  # set False for alignments converging faster
    cross_entropy_pos_weight=20,  # Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1=disabled)
    mel_loss='mae',
    spec_loss='mae',


    # Tacotron Training
    # Reproduction seeds
    random_seed=2020,  # Determines initial graph and operations (i.e: model) random state for reproducibility
    # tacotron_data_random_state=1234,  # random state for train test split repeatability

    # performance parameters
    tacotron_swap_with_cpu=False,  # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

    # train/test split ratios, mini-batches sizes
    batch_size=32,  # number of training samples on each training steps
    # Tacotron Batch synthesis supports ~16x the training batch size (no gradients during testing).
    # Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
    tacotron_synthesis_batch_size=1,  # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
    tacotron_test_size=0.05,  # % of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is enough to have a good idea about overfit)
    tacotron_test_batches=None,  # number of test batches.

    # Learning rate schedule
    decay_learning_rate=True,  # boolean, determines if the learning rate will follow an exponential decay
    start_decay=40000,  # Step at which learning decay starts
    decay_steps=18000,  # Determines the learning rate decay slope (UNDER TEST)
    decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    # initial_learning_rate=1e-3,  # starting learning rate
    initial_learning_rate=0.002,
    final_learning_rate=1e-4,  # minimal learning rate

    # Optimization parameters
    adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter

    # Regularization parameters
    # reg_weight=1e-6,  # regularization weight (for L2 regularization)
    reg_weight=None,  # regularization weight (for L2 regularization)
    scale_regularization=False,  # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)
    zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet
    clip_gradients=True,  # whether to clip gradients
)
