# -*- coding: utf-8 -*-
import re

from .base import BaseConfig


class PeterConfig(BaseConfig):

    def __init__(
            self,
            data_dir,
            image_w=2048,
            image_h=128,
            dataset_name='peter',
            chars=' #()+0123456789[]bdfghilmnrstwабвгдежзийклмнопрстуфхцчшщъыьэюяѣ⊕⊗',
            blank='ß',
            **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            image_w=image_w,
            image_h=image_h,
            chars=chars,
            blank=blank,
            **kwargs,
        )

    @staticmethod
    def preprocess(text):
        """ Метод чистки текста перед self.encode  """
        eng2rus = {
            'o': 'о',
            'a': 'а',
            'c': 'с',
            'e': 'е',
            'p': 'р',
            '×': 'х',
            '/': '',
            '…': '',
            '|': '',
            '–': '',
            'ǂ': '',
            'u': 'и',
            'k': 'к',
            'і': 'i',
        }
        text = text.strip()
        text = ''.join([eng2rus.get(char, char) for char in text])
        text = re.sub(r'\b[pр]s\b', 'р s', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def postprocess(text):
        """ Метод чистки текста после self.decode  """
        text = text.strip()
        text = text.replace('р s', 'p s')
        text = text.replace('рs', 'p s')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
