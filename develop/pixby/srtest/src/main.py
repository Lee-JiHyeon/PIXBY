import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
# import subprocess

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    # subprocess.call('--data_test Demo --scale 4 --pre_train ../experiment/edsr_baseline_x4/model/model_best.pt --test_only --ave_results --chop', shell=True)
    # print(args.data_test)
    # print(args.test_only)

    args.data_test = ['Demo']
    args.test_only = True
    args.scale = [2]
    args.pre_train = '../experiment/edsr_baseline_x2/model/model_best.pt'
    # args.save_result = True
    args.save_results = True
    args.chop = True
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test() 
    else:
        if checkpoint.ok:
            print('1111111111111111')
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
