import os
from pixby.srtest.src.data import srdata

class TESTDATA(srdata.SRData):
    def __init__(self, args, name='TESTDATA', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]    
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(TESTDATA, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(TESTDATA, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        #print(names_hr, '===hr====')
        #print(names_lr, '===lr====')

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'TESTDATA')
        self.dir_hr = os.path.join(self.apath, 'TESTDATA_train_HR')
        self.dir_lr = os.path.join(self.apath, 'TESTDATA_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')