from argparse import ArgumentParser


def IO(parser: ArgumentParser) -> None:
    """
    Add to the parser generals and input/output parameters.
    """
    parser.add_argument(
        '--dataset',
        type=str,
        help='Absolute path for msc dataset'
    )
    parser.add_argument(
        '--models_folder',
        type=str,
        help='Absolute path of the folder for saving the models.'
    )
    parser.add_argument(
        '--tflite_models_folder',
        type=str,
        help='Absolute path of the folder for saving tflite models'
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        help='Absolute path of the folder for saving csv results.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random state.'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Use this flag to flush models, tflite_models and results folders.'
    )


def preprocess(parser: ArgumentParser) -> None:
    """
    Add to the parser preprocessing parameters.
    """
    parser.add_argument(
        '--preprocess',
        type=str,
        default='mfccs',
        help='Type of preprocess, str in ["spect", "log_mel_spect", "mfccs"]'
    )
    parser.add_argument(
        '--frame_length_in_s',
        type=float,
        default=0.04,
        help='The frame length for STFT.'
    )
    parser.add_argument(
        '--frame_step_in_s',
        type=float,
        default=0.02,
        help='The step size for STFT.'
    )
    parser.add_argument(
        '--num_mel_bins',
        type=int,
        default=40,
        help='The number of bins for the mel spectrogram.'
    )
    parser.add_argument(
        '--lower_frequency',
        type=int,
        default=20,
        help='The lower frequency for the mel spectrogram.'
    )
    parser.add_argument(
        '--upper_frequency',
        type=int,
        default=4000,
        help='The upper frequency for the mel spectrogram.'
    )
    parser.add_argument(
        '--num_coefficients',
        type=int,
        default=40,
        help='The number of coefficients in mfccs.'
    )


def pruning(parser: ArgumentParser) -> None:
    """
    Add to the parser pruning parameters.
    """
    parser.add_argument(
        '--prune',
        action='store_true',
        help='Use this flag to perform weights pruning'
    )
    parser.add_argument(
        '--initial_sparsity',
        type=float,
        default=0.2,
        help='Parameter for weights pruning.'
    )
    parser.add_argument(
        '--final_sparsity',
        type=float,
        default=0.7,
        help='Parameter for weights pruning.'
    )


def training(parser: ArgumentParser) -> None:
    """
    Add to the parser training parameters.
    """
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='The size for the batches.'
    )
    parser.add_argument(
        '--initial_learning_rate',
        type=float,
        default=0.01,
        help='The initial learning rate value.'
    )
    parser.add_argument(
        '--end_learning_rate',
        type=float,
        default=1e-5,
        help='The final learning rate value.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='The number of epochs.'
    )
