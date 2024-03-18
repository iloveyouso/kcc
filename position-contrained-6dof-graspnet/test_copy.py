from options.train_options import TrainOptions
from models import create_model
from test import run_test 
from data import DataLoader
from utils.writer import Writer

def main():
    print('Running Test by BJKIM From the scratch')
    
    opt = TrainOptions().parse()
    if opt == None:
        return

    data_loader = DataLoader(opt)
    training_dataset, test_dataset, _ = data_loader.split_dataset(opt.dataset_split_ratio)

    dataset_train = data_loader.create_dataloader(training_dataset, shuffle_batches=not opt.serial_batches)
    dataset_test = data_loader.create_dataloader(test_dataset, shuffle_batches=False)
    dataset_train_size = len(training_dataset)
    dataset_test_size = len(test_dataset)
    print('#train images = %d' % dataset_train_size)
    print('#test images = %d' % dataset_test_size)


    writer = Writer(opt)

    
    run_test(epoch=-1, name=opt.name, writer=writer, dataset_test=dataset_test)

if __name__ == '__main__':
    main()
