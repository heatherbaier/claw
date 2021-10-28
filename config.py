import argparse


arg_lists = []
parser = argparse.ArgumentParser(description = "a3cRAM")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


training_args = add_argument_group("Training Arguments")
training_args.add_argument("--batch_size", 
                           type = int, 
                           default = 6, 
                           help = "batch size is equivalent to number of process to run asychronously / nodes * ppn")
training_args.add_argument("--n_epochs", 
                           type = int, 
                           default = 20, 
                           help = "Number of epochs")
training_args.add_argument("--n_mini_epochs", 
                           type = int, 
                           default = 20, 
                           help = "Number of mini epochs performed on each image within an epoch")
training_args.add_argument("--eps_decay", 
                           type = int, 
                           default = 150, 
                           help = "Value by which to decay epsilon")


env_args = add_argument_group("RL Environment Arguments")
env_args.add_argument("--display", 
                      type = str, 
                      default = "False", 
                      help = "Whether to display the interactive RL environment during training - Cannot display on HPC")


data_args = add_argument_group("Data Arguments")
data_args.add_argument("--imagery_dir",
                      type = str,
                      default = "/sciclone/home20/hmbaier/claw/train_ims",
                      help = "Full path to directory containing imagery")
data_args.add_argument("--json_path",
                      type = str,
                      default = "/sciclone/home20/hmbaier/claw/migration_data.json",
                      help = "Full path to json containing muni_id -> num_migrants mapping.")