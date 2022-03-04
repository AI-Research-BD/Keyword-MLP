import json
from argparse import ArgumentParser
from utils.dataset import get_train_val_test_split
import os
import glob
import numpy as np
import shutil
import soundfile as sf


def delete_files_from_list(file_list):
    for path in file_list:
        os.remove(path)


def make_unknown(data_list, unknown_perc, silence_perc, label_map, root):
    known_list, unknown_list = [], []
    for file_path in data_list:
        if file_path.split("/")[-2] in label_map.values():
            known_list.append(file_path)
        else:
            unknown_list.append(file_path)
    
    known_perc = 1 - (unknown_perc + silence_perc)
    num_total_data = int(len(known_list) / known_perc)
    num_unknown = int(num_total_data * unknown_perc)

    unknown_list = np.random.choice(unknown_list, num_unknown, replace=False).tolist()
    renamed_unknown_list = []

    for file_path in unknown_list:
        classname = file_path.split("/")[-2]
        new_path = os.path.join(root, "_unknown_", classname + "_" + os.path.basename(file_path))
        renamed_unknown_list.append(new_path)
        shutil.move(file_path, new_path)
    
    return known_list + renamed_unknown_list, num_total_data


def create_silence(root, stride = 2000):
    for bg_file in glob.glob(os.path.join(root, "_background_noise_", "*.wav")):
        y, sr = sf.read(bg_file)
        for i in range(0, len(y) - sr, stride):
            out_path = os.path.join(root, "_silence_", f"{os.path.basename(bg_file)[:-4]}_{i}.wav")
            sf.write(out_path, y[i: i + sr], sr)


def add_silence(train_list, val_list, total_train_size, total_val_size, root):
    num_silence_train = total_train_size - len(train_list)
    num_silence_val = total_val_size - len(val_list)

    all_silence = glob.glob(os.path.join(root, "_silence_", "*.wav"))
    train_val = np.random.choice(all_silence, num_silence_train + num_silence_val, replace=False).tolist()
    
    to_be_removed = list(set(all_silence) - set(train_val))
    delete_files_from_list(to_be_removed)

    train_list = train_list + train_val[:num_silence_train]
    val_list = val_list + train_val[num_silence_train:]

    return train_list, val_list



def main(args):

    val_list_path = os.path.join(args.root, "validation_list.txt")
    test_list_path = os.path.join(args.root, "testing_list.txt")

    train_list, val_list, test_list, _ = get_train_val_test_split(args.root, val_list_path, test_list_path)

    with open("data_prep/label_maps/v1_12.json", "r") as f:
        label_map = json.load(f)

    os.makedirs(os.path.join(args.root, "_silence_"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "_unknown_"), exist_ok=True)

    train_list, total_train_size = make_unknown(train_list, args.unknown_perc, args.silence_perc, label_map, args.root)
    val_list, total_val_size = make_unknown(val_list, args.unknown_perc, args.silence_perc, label_map, args.root)
    print("Created _unknown_ for train and val.")
    # print(total_train_size, len(train_list))
    # print(total_val_size, len(val_list))

    create_silence(args.root)
    print(f"Created {len(os.listdir(os.path.join(args.root, '_silence_')))} silence files.")

    train_list, val_list = add_silence(train_list, val_list, total_train_size, total_val_size, args.root)
    print("Added silence to train and val.")

    delete_files_from_list(test_list)
    print("Deleted test_list files.")

    for classdir in glob.glob(os.path.join(args.root, "*")):
        if os.path.isdir(classdir) and classdir.split("/")[-1] not in label_map.values():
            shutil.rmtree(classdir)
            
    print("Deleted unneeded unknown class folders.")

    test_list = []

    for filename in glob.glob(os.path.join(args.test, "*", "*.wav")):
        classname, basename = filename.split("/")[-2:]
        new_path = os.path.join(args.root, classname, basename)
        test_list.append(new_path)
        shutil.move(filename, new_path)
    print("Moved test files to root")

    with open(os.path.join(args.root, "training_list.txt"), "w+") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(args.root, "validation_list.txt"), "w+") as f:
        f.write("\n".join(val_list))

    with open(os.path.join(args.root, "testing_list.txt"), "w+") as f:
        f.write("\n".join(test_list))
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Base dataset directory.")
    parser.add_argument("--test", type=str, required=True, help="Test dataset directory.")
    parser.add_argument("--unknown_perc", type=float, default=0.1, help="Unknown percentage.")
    parser.add_argument("--silence_perc", type=float, default=0.1, help="Silence percentage.")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)