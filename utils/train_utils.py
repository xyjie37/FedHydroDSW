from torchvision import datasets, transforms
from models.lstm import LSTM, BiLSTM
import torch
import torch.nn as nn

def get_model(args):
    if args.model == 'lstm':
        lstm = LSTM(input_size=args.input_size, hidden_size=args.hidden_size,
                    num_layers=args.num_layers, output_size=args.output_size)
        net_glob = lstm.to(args.device)
    elif args.model == 'bilstm':
        lstm = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size,
                    num_layers=args.num_layers, output_size=args.output_size)
        net_glob = lstm.to(args.device)
    else:
        exit('Error: unrecognized model')

    return net_glob