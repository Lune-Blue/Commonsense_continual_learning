import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    
def parser_args():
    parser = argparse.ArgumentParser()
    ## root dir = os.path.expanduser('~')
    parser.add_argument('--path', type=str, default='/home/intern/nas/results/seungjun')
    parser.add_argument('--condition', type=str, default='kd')
    parser.add_argument('--include', type=str2bool, default="True")
    args = parser.parse_args()
    return args


def get_result(path, condition, includes):
    directory = os.listdir(path)
    get_path = [os.path.join(path, get_directory) for get_directory in directory]
    all_files=[]
    for files in get_path:
        get_new_files = os.listdir(files)
        get_directory_files = [os.path.join(files,file) for file in get_new_files]
        all_files.extend(get_directory_files)
    if includes:    
        files = [file for file in all_files if (condition in file)]
    else:
        files = [file for file in all_files if (condition not in file)]
    acc_arr=[]
    for file in files:
        f = open(file, 'r')
        acc_arr.append(f.readline())
    if includes:
        name = '{}-includes.txt'.format(condition)
    else:
        name = '{}-not-includes.txt'.format(condition)
    w = open(os.path.join(os.getcwd(),name),'w')
    for acc in acc_arr:
        w.write(acc)

def main(args):
    get_result(args.path, args.condition, args.include)
    
if __name__ == '__main__':
    args = parser_args()
    main(args)