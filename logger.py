class Logger(object):
    '''Save training process log '''

    def __init__(self, fpath, resume=False):
        self.file = None
        self.resume = resume
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()                                                                         #read the top line 'name'
                self.names = name.rstrip().split()
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split()
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])                                               #
                self.file.close()
                self.file = open(fpath, 'a')                                                                         #a to restart
            else:
                self.file = open(fpath, 'w')                                                                          #not restart write the log.txt

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty dict for record
        self.numbers = {}                                                                                           #list in dict
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []                                                                                    #empty value at start                                                                         
        self.file.write('\n')                                                                                          #first-in write top line title
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if type(num) == str:
                self.file.write("{0:>10}".format(num))
            else:
                self.file.write("{0:>10.6f}".format(num))
            self.file.write('\t')                                                                                      
            self.numbers[self.names[index]].append(num)                                                                #append record
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()






