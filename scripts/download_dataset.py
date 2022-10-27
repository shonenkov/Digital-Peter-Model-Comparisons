# -*- coding: utf-8 -*-
import os
import tarfile
import argparse

import gdown


DATASETS_ID = {
    'bentham': '13S6lwxuFoM1vOBlofK8VUnQ3RKC2iFI0',
    'peter': '1DWQS-7RNJ1AM02lTDBAqM94tvuVRYFx_',
    'iam': '1jH81RSQfFVnRlipu3_tdICzGiRNjGLE5',
    'saintgall': '1AVidj-ZKhLKuRQCnlIAOuTf4CvKYMNMr',
}


def extract_archive(archive_path, extract_path):
    if archive_path.endswith('tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, path=extract_path)


def download_and_extract(data_dir, dataset_name):
    archive_path = f'{data_dir}/{dataset_name}.tar.gz'
    if not os.path.exists(archive_path):
        gdown.download(f'https://drive.google.com/uc?id={DATASETS_ID[dataset_name]}', archive_path, quiet=False)

    if not os.path.exists(f'{data_dir}/{dataset_name}'):
        extract_archive(archive_path, data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset script.')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--data_dir', type=str, default='../input')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    download_and_extract(args.data_dir, args.dataset_name)
