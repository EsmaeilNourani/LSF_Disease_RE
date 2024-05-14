import numpy as np
class Negative_Downsampler(object):
    def __init__(self, lp, program_halt, configs, downsampling_rate , training_set) :
        self.lp = lp
        self.program_halt = program_halt
        self.configs = configs
        self.__downsampling_rate = int(downsampling_rate)
        self.__return_original_data = False

        if not (0 <= downsampling_rate <= 100):
            self.program_halt("downsampling_rate shound be an integer in range[0,100], given: " + str(downsampling_rate))

        self.__trainset_x = training_set['x']
        self.__trainset_y = training_set['y_true']

        if self.__downsampling_rate == 0:
            self.lp("given downsampling_rate ==0 , hence no negative down_sampling will be performed. Will use all available data. Exiting negative downsampling...")
            self.__return_original_data = True
            return

        # calculate total example count
        y = self.__trainset_y[0]  # assuming the first matrix in y_true is always relation-types classification.
        self.__total_example_count = y.shape[0]

        if self.configs['classification_type'] == 'multi-label':
            """
            if there is any nonzero element, it is a positive
            y = np.array ([[0, 0, 0],
                           [0, 1, 1],
                           [0, 0, 1],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
            --> __pos_indices --> array([ 1,  2,  5, 10])
            --> __neg_indices --> array([ 0,  3,  4,  6,  7,  8,  9, 11])
            """
            self.__pos_neg_true_false_array = np.any(y == 1, axis=1)  # --> array([ True,  True,  True, ..., False, True, False]), AND __pos_neg_true_false_array.shape --> (total_example_count,)
            self.__pos_indices = np.where(self.__pos_neg_true_false_array == True)[0]  # --> array([0,1, 2, ..., 63332])
            self.__neg_indices = np.where(self.__pos_neg_true_false_array == False)[0]  # --> array([100, 102, ..., 63454])

        else:  # 'binary' and 'multi-class' use softmax and the first element shows if it is a negative or not. Example: [1,0,0] --> negative , [0,1,0] --> positive
            """separate positives from negatives
             y= array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0]])

            y[:, 0] --> array([1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1])
            pos_neg_true_false_array = y[:, 0]
            negatives: np.where(pos_neg_true_false_array == 1)[0] --> array([ 0,  3,  4,  6,  7,  8,  9, 11])
            positives: np.where(pos_neg_true_false_array == 0)[0] --> array([ 1,  2,  5, 10]) 
            """
            self.__pos_neg_true_false_array = y[:, 0]
            self.__pos_indices = np.where(self.__pos_neg_true_false_array == 0)[0]
            self.__neg_indices = np.where(self.__pos_neg_true_false_array == 1)[0]

        self.__total_pos_count = self.__pos_indices.shape[0]
        self.__total_neg_count = self.__neg_indices.shape[0]

        msg = ["-" * 32 + " NEGATIVE DOWNSAMPLING " + "-" * 32]
        msg += ["total examples  count : " + str(self.__total_example_count)]
        msg += ["total positives count : " + str(self.__total_pos_count)]
        msg += ["total negatives count : " + str(self.__total_neg_count)]

        if self.__total_neg_count == 0:
            msg += ["[WARNING] NO NEGATIVES FOUND! ... Will use available data. Exiting negative downsampling..."]
            self.lp(msg)
            self.__return_original_data = True
            return

        if self.__total_neg_count <= self.__total_pos_count:
            msg += ["total_neg_count <= total_pos_count , hence will use all available data. Exiting negative downsampling..."]
            self.lp(msg)
            self.__return_original_data = True
            return

        try:
            assert self.__total_example_count == self.__total_pos_count + self.__total_neg_count
        except Exception as E:
            msg += ["Inconsistency in calculations in negative downsampling ! Halting"]
            program_halt(msg)

        self.__neg_pos_difference = self.__total_neg_count - self.__total_pos_count
        self.__downsampling_value = int((self.__downsampling_rate * self.__neg_pos_difference) / 100.0)
        self.__new_negative_count = self.__total_neg_count - self.__downsampling_value

        msg += ["negatives - positives : " + str(self.__neg_pos_difference)]
        msg += ["downsampling_rate     : " + str(self.__downsampling_rate)]
        msg += ["downsampling_value    : " + str(self.__downsampling_value)]
        msg += ["new_negatives_count   : " + str(self.__new_negative_count)]
        msg += ["-" * 85]
        lp(msg)

    def get_new_sample(self):
        if self.__return_original_data:
            return self.__trainset_x, self.__trainset_y

        # how to get a sample without replacement : np.random.choice(neg_indices, new_negative_count, replace=False)
        # how to get some elements : np.array(train_set_x[0])[pos_indices]
        neg_sub_indices = np.random.choice(self.__neg_indices, self.__new_negative_count, replace=False)
        selected_indices = np.hstack((self.__pos_indices, neg_sub_indices))
        np.random.shuffle(selected_indices)
        subsample_x = []
        subsample_y = []
        for matrix in self.__trainset_x:
            subsample_x.append(np.array(matrix)[selected_indices])
        for matrix in self.__trainset_y:
            subsample_y.append(np.array(matrix)[selected_indices])
        return subsample_x ,subsample_y
