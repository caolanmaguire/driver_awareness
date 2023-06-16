import logging

logging.basicConfig(filename = 'report.log',level = logging.DEBUG,format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logging.info('Info message')