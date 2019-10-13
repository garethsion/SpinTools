import logging

class Logger:
    def logger(self,filename):
        """ This is a logger which aids debugging during code development"""
        LOG_FORMAT = "%(Levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(filename = filename + ".Log",
                level = logging.DEBUG,
                format = LOG_FORMAT,
                filemode = 'w')
        # logger = logging.getLogger()
        return 
