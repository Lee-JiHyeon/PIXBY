import os
import math
from decimal import Decimal

from pixby.srtest.src import utility

import torch
import torch.nn.utils as utils
# from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, window):
        self.args = args
        self.scale = args.scale
        self.window = window

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)), self.window
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        # print(self.loader_train.num_workers, 'self.loader_tran===================')
        # print((self.loader_train), '갯수')
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            # 0514 내가 추가한 곳 0515 자리 이동-------------------
            lr = lr[:, :3, :, :]
            hr = hr[:, :3, :, :]
            # print(lr, hr, 'lr hr')
            # ---------------------------------------
            #print(lr, hr, '================lr hr================')
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()),
                    self.window)

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
      
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:', self.window)
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        # 0514 multiprocessing 지우려고 주석처리-> 했다가 품
        # issue 105확인
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):     
            with torch.no_grad():
                for idx_scale, scale in enumerate(self.scale): 
                    d.dataset.set_scale(idx_scale)
                    _nums = len(d.dataset)
                    _cnt = 1
                  
                    for lr, hr, filename in d:
                       
                        lr, hr = self.prepare(lr, hr)
                        lr = lr[:, :3, :, :]
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        
                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        
                        if self.args.save_gt:
                            save_list.extend([lr, hr])
                        
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        
                        _per = _cnt * 100 // _nums
                        _tpm = str(filename) + ' 변환중입니다...   : ' + str(_per) +  '/  100  %'

                        self.window.textBox_terminal.append(_tpm)
                        _cnt += 1
                        # 2번째 찍는곳 찾는중
                        # print("두번쨰")

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        ),
                        self.window
                    )
                    #  여기에 완료창 표시해주기
                    


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()), self.window)
        self.ckp.write_log('Saving...', self.window)

        # 0514 multiprocessing 지우려고 주석처리했다가 주석 품
        # issue 105확인
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, self.window, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), self.window, refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

