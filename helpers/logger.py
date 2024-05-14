import os
import sys
import inspect

from helpers import general_helpers as GH

class Logger:
    def __init__(self, file_path, allow_append_if_file_exists=False, write_header_to_file=True):
        self.__file_handle = None

        if not isinstance(file_path, str):
            print("argument file_path should be string.")
            sys.exit(-1)

        if not isinstance(allow_append_if_file_exists, bool):
            print("argument allow_append_if_file_exists should be bool.")
            sys.exit(-1)

        if os.path.isfile(file_path):
            if not allow_append_if_file_exists:
                print ("[ERROR] log file already exists and append is not allowed. Exiting program!\nfile: " + file_path)
                sys.exit(-1)
            else:
                print("[WARNING] log file already exists and append is allowed. All new logs will be appended to the current log file!\nfile: " + file_path)
                try:
                    self.__file_handle = open(file_path, "w") #TODO: change to a later ...
                    print("successfully opened the log file for append.")
                except Exception as E:
                    print ("Error trying to open the log file.\nFile: " + file_path + "\nError:" + str(E))
                    sys.exit(-1)
        else:
            try:
                print ("Creating new log file: " + file_path)
                self.__file_handle = open(file_path, "wt")
                print("successfully created and opened the log file.")
            except Exception as E:
                print ("Error trying to create and open the log file.\nFile: " + file_path + "\nError:" + str(E))
                sys.exit(-1)

        if write_header_to_file:
            self.lp(["="* 140, "PROGRAM START.", "="* 140 ])

    def is_open(self):
        return (self.__file_handle is not None) and (not self.__file_handle.closed)

    def close(self):
        if self.is_open():
            try:
                self.__file_handle.flush()
                self.__file_handle.close()
                print("log file closed successfully.")
                self.__file_handle = None
            except Exception as E:
                print("Error in closing the log file.\Error:" + str(E))

    def lp (self, message): #log and print message
        """
        to get current function name: inspect.stack()[0][3]  or: inspect.currentframe().f_code.co_name
        to get caller  function name: inspect.stack()[1][3]  or: inspect.currentframe().f_back.f_code.co_name  #second does not work for decorated method !
        """
        try:
            _caller_class_name = str(inspect.stack()[1][0].f_locals["self"].__class__)
        except:
            _caller_class_name = ""

        _caller_function_name = inspect.currentframe().f_back.f_code.co_name
        if isinstance(message, str):
            message = [message]
        _msg = "[" + GH.datetime_get_now() + "] [" + _caller_class_name + "." + _caller_function_name + "]: "
        print (_msg)
        self.__file_handle.write (_msg+"\n")
        for itemstr in message:
            try:
                itemstr = str(itemstr).replace ('\r', '\n')
            except:
                itemstr = itemstr.encode('utf-8').replace ('\r', '\n')

            for item in itemstr.split ("\n"):
                if len(item)==0:
                    item = "-"
                item = "      "+item
                print (item)
                self.__file_handle.write (item+"\n")
        print ("")
        self.__file_handle.write ("\n")
        self.__file_handle.flush ()

    def lp_halt (self, message):
        _caller_function_name = inspect.currentframe().f_back.f_code.co_name
        if isinstance(message, str):
            self.lp (["*"*80, "HALT REQUESTED BY FUNCTION: " + _caller_function_name, "HALT MESSAGE: ", message, "HALTING PROGRAM!!!", "*"*80])
        else:
            self.lp(["*" * 80, "HALT REQUESTED BY FUNCTION: " + _caller_function_name, "HALT MESSAGE: "] + ["\t"+i for i in message] + ["HALTING PROGRAM!!!", "*" * 80])

