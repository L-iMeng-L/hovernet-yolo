# utils/logger.py
import csv
import os

class CSVLogger:
    def __init__(self, save_dir, filename='log.csv'):
        self.path = os.path.join(save_dir, filename)
        self._file= None
        self._writer = None

    def init(self, fieldnames):
        """第一次调用时创建文件并写表头"""
        is_new = not os.path.exists(self.path)
        self._file   = open(self.path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
            self._file.flush()
        self._fieldnames = fieldnames

    def log(self, row: dict):
        """写一行，缺失字段自动填 None"""
        full_row = {k: row.get(k, None) for k in self._fieldnames}
        self._writer.writerow(full_row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()