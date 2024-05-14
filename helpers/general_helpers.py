import sys
import json
import gzip
import shutil
import datetime

def nlvr (s, cnt):
    #for pretty printing s, with maximum cnt characters. If spaces are needed, they are added to the RIGHT side of the s.
    if s is None:
        return " " * cnt
    s = str(s)
    if len(s) < cnt:
        return str(s) + (" " * (cnt - len(s)))
    else:
        return s[0:cnt]

def nlvl (s, cnt):
    #for pretty printing s, with maximum cnt characters. If spaces are needed, they are added to the LEFT side of the s.
    if s is None:
        return " " * cnt
    s = str(s)
    if len(s) < cnt:
        return (" " * (cnt - len(s))) + str(s)
    else:
        return s[0:cnt]

def datetime_get_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def datetime_get_now_for_logfile():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def get_all_files_with_extension (folder_address, file_extension, process_sub_folders = True):
    #IMPORTANT ... Extenstion should be like : "txt" , "a2"  ... WITHOUT DOT !    
    all_files = []
    if process_sub_folders:
        for root, dirs, files in shutil.os.walk(folder_address):
            for file in files:
                if file.endswith("." + file_extension):
                    all_files.append(shutil.os.path.join(root, file))
        return (all_files)
    else:         
        for file in shutil.os.listdir(folder_address):
            if file.endswith("." + file_extension): #".txt" ;
                all_files.append(folder_address + file)
        return (all_files)

def get_immediate_subdirectories(path , return_full_path=True):
    #return_full_path == True --> returns path to each subfolder
    #return_full_path == False --> returns only subfolders' names
    if return_full_path:
        return [f.path for f in shutil.os.scandir(path) if f.is_dir()]
    else:
        return [f.name for f in shutil.os.scandir(path) if f.is_dir()]

def float_decimal_points (num, n):
    #pretty-print for floating point numbers with exactly N digits after .
    r = "{:." + str(n) + "f}"
    return r.format (num)

def partition_into_exactly_n_partitions(L, N):
    if N > len(L):
        print ("Error in LIST_PartitionIntoNPartitions: N should be in [1,len(L)]")
        sys.exit(-1)
    res = []
    for i in range (0, N):
        res.append ([])
    for i in range(0, len(L)):
        res[i % N].append (L[i])
    return res

def read_gziped_json(file_address):
    with gzip.GzipFile(file_address, 'r') as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    return data

def OS_mkdir (folder_address):
    if not shutil.os.path.exists(folder_address):
        try:
            shutil.os.makedirs(folder_address)
            print(("Creating Folder: ", folder_address))
        except: #Exception handling and passing is needed for multi-threaded execution since it raises error if folder already exists.
            pass

def OS_rm_directory_with_content(folder_address):
    if shutil.os.path.exists(folder_address):
        shutil.rmtree(folder_address)

def OS_check_folder_exists(folder_address, should_have_subfolders=False, should_have_immediate_files=False):
    if folder_address[-1] != "/":
        folder_address+= "/"
    if not shutil.os.path.exists(folder_address):
        return False, folder_address, "folder not found: " + folder_address
    if should_have_subfolders:
        has_immediate_folders = len([name for name in shutil.os.listdir(folder_address) if shutil.os.path.isdir(shutil.os.path.join(folder_address, name))])
        if not has_immediate_folders:
            return False, folder_address, "the folder does not have any sub-folders inside."
    if should_have_immediate_files:
        has_immediate_files = len([name for name in shutil.os.listdir(folder_address) if shutil.os.path.isfile(shutil.os.path.join(folder_address, name))])
        if not has_immediate_files:
            return False, folder_address, "the folder does not have any immediate files inside."
    return True, folder_address, ""