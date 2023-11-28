# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 封装的一个日志类
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2019-10-24 13:32:14
 LastEditors  : huzhenhong
 LastEditTime : 2020-10-14 20:56:25
 FilePath     : \\image_algorithm\\base_library\\logger.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import os
import logging
from logging import handlers


class Logger(object):
    def __init__(
                self,
                filename,
                level='WARNING',
                when='D',
                backCount=3,
                fmt='[%(asctime)s]-[%(filename)s-line:%(lineno)d]-[%(funcName)s]-[%(levelname)s]: %(message)s'):
        # fmt='%(asctime)s-p:%(process)d-t:%(thread)d-%(filename)s-line:%(lineno)d-%(funcName)s-%(levelname)s: %(message)s'):

        abs_path = os.path.realpath(filename)
        path = abs_path[0: abs_path.rfind('/')]

        if not os.path.exists(path):
            os.makedirs(path)

        self.logger = logging.getLogger(abs_path)               # 根据文件名创建一个日志
        self.logger.setLevel(level)                             # 设置默认日志级别
        self.format_str = logging.Formatter(fmt)                # 设置日志格式

        screen_handler = logging.StreamHandler()                # 屏幕输出处理器
        screen_handler.setFormatter(self.format_str)            # 设置屏幕输出显示格式

        # 定时写入文件处理器
        time_file_handler = handlers.TimedRotatingFileHandler(filename=filename,        # 日志文件名
                                                              when=when,                # 多久创建一个新文件
                                                              interval=1,               # 写入时间间隔
                                                              backupCount=backCount,    # 备份文件的个数
                                                              encoding='utf-8')         # 编码格式

        time_file_handler.setFormatter(self.format_str)

        # 添加日志处理器
        self.logger.addHandler(screen_handler)
        self.logger.addHandler(time_file_handler)

    def loggerImp(self):
        return self.logger
