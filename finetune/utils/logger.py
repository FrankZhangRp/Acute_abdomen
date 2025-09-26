import os
import logging

def get_logger(file_name, file_save=True, display=True):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    
    logger = logging.Logger(file_name, logging.INFO)
    logger.setLevel(logging.INFO)
    if file_save:
        if os.path.isfile(file_name):
            fh = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        else:
            fh = logging.FileHandler(file_name, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if display:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger