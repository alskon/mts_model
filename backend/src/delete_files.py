import os


def del_files(path):
    try:
        files = os.listdir(path)
        print(files)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print('Files deleted')
    except OSError:
        pass