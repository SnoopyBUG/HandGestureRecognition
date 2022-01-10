import os
import os.path


def write_txt(content, filename, mode="w"):
    """
    The meaning of the parameter:
            content: the data needed to save, type->list
            filename: the name of your file
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if(not col == len(line)-1):
                    # Using the Space to divide
                    str_line = str_line + str(data) + " "
                else:
                    # The last data each line, using "\n"
                    str_line = str_line + str(data)+"\n"
            f.write(str_line)


def get_files_list(dir):
    """
    go through all the files under the dir 
    :param dir: the route
    :return: a list containing all the files->list
    """
    # parent: parent route, dirnames: all the folders under the route,filenames:all the files under the route
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            print("parent is: " + parent)
            print("filename is: " + filename)
            # output the information of all the files
            print(os.path.join(parent, filename).replace('\\', '/'))
            curr_file = parent.split(os.sep)[-1]  # get the name of class
            # define the label according to the name of class
            labels = eval(curr_file) - 1
            dir_path = parent.replace('\\', '/').split('/')[-2]  # train?val?test?
            # route(相对路径)
            curr_file = os.path.join(dir_path, curr_file)
            files_list.append([os.path.join(curr_file, filename).replace('\\', '/'), labels])  # route+label

            # write into the csv
            path = "%s" % os.path.join(curr_file, filename).replace('\\', '/')
            label = "%d" % labels
            list = [path, label]
            data = pd.DataFrame([list])
            if(dir == './HandDataset/train'):
                data.to_csv("./HandDataset/train.csv", mode='a', header=False, index=False)
            elif(dir == './HandDataset/test'):
                data.to_csv("./HandDataset/test.csv", mode='a', header=False, index=False)
    return files_list


if __name__ == '__main__':
    import pandas as pd
    # 以下用于创建csv文件
    # df = pd.DataFrame(columns=['path', 'label'])
    # df.to_csv("./HandDataset/train.csv", index=False)

    # df2 = pd.DataFrame(columns=['path', 'label'])
    # df2.to_csv("./HandDataset/test.csv", index=False)


    # 创建train和test的文件
    train_dir = 'IBMDvsGesture/4x100ms-Ignore/IBMDvs_30/train'
    train_txt = 'IBMDvsGesture/4x100ms-Ignore/IBMDvs_30/train/path-label_train.txt'
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode="w")

    test_dir = 'IBMDvsGesture/4x100ms-Ignore/IBMDvs_30/test'
    test_txt = 'IBMDvsGesture/4x100ms-Ignore/IBMDvs_30/test/path-label_test.txt'
    test_data = get_files_list(test_dir)
    write_txt(test_data, test_txt, mode="w")
