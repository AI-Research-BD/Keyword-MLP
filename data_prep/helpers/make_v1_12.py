import json
from argparse import ArgumentParser

from sklearn.utils import shuffle
from utils.dataset import get_train_val_test_split
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
import shutil
import soundfile as sf


def delete_files_from_list(file_list: list) -> None:
    """Deletes files from a list of file paths.

    Args:
        file_list (list): File paths.
    """
    for path in file_list:
        os.remove(path)


def make_unknown(
    data_list: list,
    unknown_perc: float,
    silence_perc: float,
    label_map: dict,
    root: str,
):
    """Produces unknown.

    Args:
        data_list (list): List of files.
        unknown_perc (float): Percentage of unknown in range [0., 1.]
        silence_perc (float): Percentage of silence in range [0., 1.]
        label_map (dict): Index to class mapping.
        root (str): Dataset root directory, where new _unknown_ folder will be made.

    Returns:
        data_list (list): List of known class files + unknown class files (their new paths under _unknown_).
        num_total_data (int): Projected size of full dataset with silence and unknown samples.
    """

    known_list, unknown_list, unknown_class_list = [], [], []
    for file_path in data_list:
        classname = file_path.split("/")[-2]
        if classname in label_map.values():
            known_list.append(file_path)
        else:
            unknown_list.append(file_path)
            unknown_class_list.append(classname)

    known_perc = 1 - (unknown_perc + silence_perc)
    num_total_data = int(len(known_list) / known_perc)  # Expected size of whole dataset
    num_unknown = int(num_total_data * unknown_perc)
    unknown_frac = num_unknown / len(
        unknown_list
    )  # fraction of unknown list which will go into data_list

    # use train_test_split to sample unknowns with stratification
    unknown_list, _, _, _ = train_test_split(
        unknown_list,
        unknown_class_list,
        train_size=unknown_frac,
        random_state=0,
        shuffle=True,
        stratify=unknown_class_list,
    )

    # unknown_list = np.random.choice(unknown_list, num_unknown, replace=False).tolist()
    renamed_unknown_list = []

    for file_path in unknown_list:
        classname = file_path.split("/")[-2]
        new_path = os.path.join(
            root, "_unknown_", classname + "_" + os.path.basename(file_path)
        )
        renamed_unknown_list.append(new_path)
        shutil.move(file_path, new_path)

    return known_list + renamed_unknown_list, num_total_data


def create_silence(root, train_list, val_list, total_train_size, total_val_size):
    stride = 2000
    all_silence, silence_source = [], []

    for bg_file in glob.glob(os.path.join(root, "_background_noise_", "*.wav")):
        y, sr = sf.read(bg_file)
        for i in range(0, len(y) - sr, stride):
            out_path = os.path.join(
                root, "_silence_", f"{os.path.basename(bg_file)[:-4]}_{i}.wav"
            )
            sf.write(out_path, y[i : i + sr], sr)
            all_silence.append(out_path)
            silence_source.append(bg_file)

    num_silence_train = total_train_size - len(train_list)
    num_silence_val = total_val_size - len(val_list)

    num_silence_train_val = num_silence_train + num_silence_val
    silence_frac = num_silence_train_val / len(all_silence)

    (
        train_val_silence,
        to_be_removed_silence,
        train_val_silence_source,
        _,
    ) = train_test_split(
        all_silence,
        silence_source,
        train_size=silence_frac,
        random_state=0,
        shuffle=True,
        stratify=silence_source,
    )
    silence_train_frac = num_silence_train / len(train_val_silence)
    train_silence, val_silence, _, _ = train_test_split(
        train_val_silence,
        train_val_silence_source,
        train_size=silence_train_frac,
        random_state=0,
        shuffle=True,
        stratify=train_val_silence_source,
    )

    delete_files_from_list(to_be_removed_silence)
    return train_list + train_silence, val_list + val_silence


# def add_silence(train_list, val_list, total_train_size, total_val_size, root):
#     num_silence_train = total_train_size - len(train_list)
#     num_silence_val = total_val_size - len(val_list)

#     all_silence = glob.glob(os.path.join(root, "_silence_", "*.wav"))
#     train_val = np.random.choice(all_silence, num_silence_train + num_silence_val, replace=False).tolist()

#     to_be_removed = list(set(all_silence) - set(train_val))
#     delete_files_from_list(to_be_removed)

#     train_list = train_list + train_val[:num_silence_train]
#     val_list = val_list + train_val[num_silence_train:]

#     return train_list, val_list


def main(args):

    val_list_path = os.path.join(args.root, "validation_list.txt")
    test_list_path = os.path.join(args.root, "testing_list.txt")

    train_list, val_list, test_list, _ = get_train_val_test_split(
        args.root, val_list_path, test_list_path
    )

    with open("data_prep/label_maps/v1_12.json", "r") as f:
        label_map = json.load(f)

    os.makedirs(os.path.join(args.root, "_silence_"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "_unknown_"), exist_ok=True)

    train_list, total_train_size = make_unknown(
        train_list, args.unknown_perc, args.silence_perc, label_map, args.root
    )
    val_list, total_val_size = make_unknown(
        val_list, args.unknown_perc, args.silence_perc, label_map, args.root
    )
    print("Created _unknown_ for train and val.")
    # print(total_train_size, len(train_list))
    # print(total_val_size, len(val_list))

    train_list, val_list = create_silence(
        args.root, train_list, val_list, total_train_size, total_val_size
    )
    print(f"Created silence files for train and val.")

    delete_files_from_list(test_list)

    print("Deleted test_list files.")

    for classdir in glob.glob(os.path.join(args.root, "*")):
        if (
            os.path.isdir(classdir)
            and classdir.split("/")[-1] not in label_map.values()
        ):
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
    parser.add_argument(
        "--root", type=str, required=True, help="Base dataset directory."
    )
    parser.add_argument(
        "--test", type=str, required=True, help="Test dataset directory."
    )
    parser.add_argument(
        "--unknown_perc", type=float, default=0.1, help="Unknown percentage."
    )
    parser.add_argument(
        "--silence_perc", type=float, default=0.1, help="Silence percentage."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
