import os
import torch

from pixby.srtest.src import utility
from pixby.srtest.src import data
from pixby.srtest.src import model
from pixby.srtest.src import loss
from pixby.srtest.src.option import args
from pixby.srtest.src.trainer import Trainer
import multiprocessing
# import subprocess



def main(window, **kwargs):
    global model
    # subprocess.call('--data_test Demo --scale 4 --pre_train ../experiment/edsr_baseline_x4/model/model_best.pt --test_only --ave_results --chop', shell=True)
    # print(args.data_test)
    # print(args)
    # print(type(args), '@#!#@!!@#!@##@!#!@#')
    # args.data_test = ['Demo']
    # args.scale = [2]
    # args.pre_train = '../experiment/edsr_baseline_x2/model/model_best.pt'
    # args.save_results = True
    # args.chop = True
    

    if torch.cuda.is_available(): 
        DEVICE = torch.device('cuda')
        
    else:
        DEVICE = torch.device('cpu')

    # print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
    # window.textBox_terminal.append("Training Done!")
    for key, value in kwargs.items():
        vars(args)[key] = value
        # parser.add_argument(f'--{key}', value)
        # args_key = f'{key}'

    torch.manual_seed(args.seed)

    checkpoint = utility.checkpoint(args)
    
        # args.args_key = value
        # print(key, value, 'key value')
        
    #     print(args.key)
    #     print(key)
    #     print(args)

    # print(args.test_only, 'asddassdasaddasasd')

    # print(args.scale, args.test_only,'asddassdasaddasasd')

    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test() 
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint, window)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()
    
    return

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    main(window, **kwargs)

