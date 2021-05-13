import subprocess

class Go():
    def __init__(self):
        data_test = 'Demo'
        scale = 4
        epoch = 300
        pre_train = 'pixby/srtest/experiment/edsr_baseline_x4/model/model_best.pt'
        tested = '--test_only'
        save = 'test'
        dir_demo = 'pixby/srtest/test'
        # dir_data = 
        subprocess.call(f'python pixby/srtest/src/main.py --data_test {data_test} --scale {scale} --pre_train {pre_train} --epoch {epoch} {tested} --save {save} --dir_demo {dir_demo} --save_results --chop', shell=True)