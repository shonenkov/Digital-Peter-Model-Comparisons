# -*- coding: utf-8 -*-
import argparse
import sys
import time
from datetime import datetime

import neptune
import torch
import numpy as np
from torch.utils.data import SequentialSampler, RandomSampler
import pandas as pd

sys.path.insert(0, '.')

from configs import CONFIGS  # noqa
from src.dataset import DatasetRetriever  # noqa
from src.converters import CTCLabeling, AttnLabeling  # noqa
from src.model import Model  # noqa
from src.experiment import OCRExperiment  # noqa
from src import utils  # noqa
from src.predictor import Predictor  # noqa
from src.metrics import string_accuracy, cer, wer  # noqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run train script.')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--experiment_description', type=str)
    parser.add_argument('--neptune_project', type=str)
    parser.add_argument('--neptune_token', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--image_w', type=int)
    parser.add_argument('--image_h', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--seed', type=int, default=6955)
    parser.add_argument('--use_progress_bar', type=int, default=0)
    parser.add_argument('--FeatureExtraction', type=str)
    parser.add_argument('--SequenceModeling', type=str)
    #
    parser.add_argument('--Transformation', type=str, default='None')
    parser.add_argument('--Prediction', type=str, default='CTC')
    parser.add_argument('--batch_max_length', type=int, default=100)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--output_channel', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    args = parser.parse_args()

    assert args.dataset_name in CONFIGS
    if args.checkpoint_path:
        seed = round(datetime.utcnow().timestamp()) % 10000  # warning! in resume need change seed
    else:
        seed = args.seed

    utils.seed_everything(seed)

    config = CONFIGS[args.dataset_name](
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        experiment_description=args.experiment_description,
        image_w=args.image_w,
        image_h=args.image_h,
        num_epochs=args.num_epochs,
        bs=args.bs,
        num_workers=args.num_workers,
        seed=seed,
        batch_max_length=args.batch_max_length,
        FeatureExtraction=args.FeatureExtraction,
        SequenceModeling=args.SequenceModeling,
        Prediction=args.Prediction,
        Transformation=args.Transformation,
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    print('DATASET:', args.dataset_name)

    if args.Prediction == 'Attn':
        converter = AttnLabeling(config)
    elif args.Prediction == 'CTC':
        converter = CTCLabeling(config)
    else:
        raise ValueError('Unknown type of prediction')

    args.num_class = len(converter.chars)
    args.imgH = args.image_h
    args.imgW = args.image_w

    df = pd.read_csv(f'{args.data_dir}/{args.dataset_name}/marking.csv', index_col='sample_id')

    train_dataset = DatasetRetriever(df[df['stage'] == 'train'], config, converter)
    valid_dataset = DatasetRetriever(df[df['stage'] == 'valid'], config, converter)
    test_dataset = DatasetRetriever(df[df['stage'] == 'test'], config, converter)

    def count_parameters(model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")
        return total_params

    model = Model(args)
    count_parameters(model)

    print(model)
    model = model.to(device)

    if args.Prediction == 'Attn':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    elif args.Prediction == 'CTC':
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        raise ValueError('Unknown type of prediction')

    optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer']['params'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['bs'],
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config['bs'],
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['bs'],
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers=config['num_workers'],
        collate_fn=utils.kw_collate_fn
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        **config['scheduler']['params'],
    )

    neptune_kwargs = {}
    if args.neptune_project:
        neptune.init(
            project_qualified_name=args.neptune_project,
            api_token=args.neptune_token,
        )
        neptune_kwargs = dict(
            neptune=neptune,
            neptune_params={
                'description': config['experiment_description'],
                'params': config.params,
            }
        )

    if not args.checkpoint_path:
        experiment = OCRExperiment(
            experiment_name=config['experiment_name'],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            base_dir=args.output_dir,
            best_saving={'cer': 'min', 'wer': 'min', 'acc': 'max'},
            last_saving=True,
            low_memory=True,
            verbose_step=10**5,
            seed=seed,
            use_progress_bar=bool(args.use_progress_bar),
            **neptune_kwargs,
            converter=converter,
            config=config,
        )
        experiment.fit(train_loader, valid_loader, config['num_epochs'])
    else:
        print('RESUMED FROM:', args.checkpoint_path)
        experiment = OCRExperiment.resume(
            checkpoint_path=args.checkpoint_path,
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs=config['num_epochs'],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            seed=seed,
            neptune=neptune_kwargs.get('neptune'),
            converter=converter,
            config=config,
        )

    time_inference = []
    for best_metric in ['best_cer', 'best_wer', 'best_acc', 'last']:
        experiment.load(f'{experiment.experiment_dir}/{best_metric}.pt')
        experiment.model.eval()
        predictor = Predictor(experiment.model, device)
        time_a = time.time()
        predictions = predictor.run_inference(test_loader)
        time_b = time.time()
        time_inference.append(time_b - time_a)
        df_pred = pd.DataFrame([{
            'id': prediction['id'],
            'pred_text': converter.decode(prediction['raw_output'].argmax(1)),
            'gt_text': prediction['gt_text']
        } for prediction in predictions]).set_index('id')

        cer_metric = round(cer(df_pred['pred_text'], df_pred['gt_text']), 5)
        wer_metric = round(wer(df_pred['pred_text'], df_pred['gt_text']), 5)
        acc_metric = round(string_accuracy(df_pred['pred_text'], df_pred['gt_text']), 5)

        if args.neptune_project:
            experiment.neptune.log_metric(f'cer_test__{best_metric}', cer_metric)
            experiment.neptune.log_metric(f'wer_test__{best_metric}', wer_metric)
            experiment.neptune.log_metric(f'acc_test__{best_metric}', acc_metric)

        mistakes = df_pred[df_pred['pred_text'] != df_pred['gt_text']]
        df_pred.to_csv(f'{experiment.experiment_dir}/pred__{best_metric}.csv')

        if args.neptune_project:
            experiment.neptune.log_metric(f'mistakes__{best_metric}', mistakes.shape[0])
            experiment.neptune.log_artifact(f'{experiment.experiment_dir}/pred__{best_metric}.csv')

        experiment._log(  # noqa
            f'Results for {best_metric}.pt.',
            cer=cer_metric,
            wer=wer_metric,
            acc=acc_metric,
            speed_inference=len(test_dataset) / (time_b - time_a),
        )

    if args.neptune_project:
        experiment.neptune.log_metric('time_inference', np.mean(time_inference))
        experiment.neptune.log_metric('speed_inference', len(test_dataset) / np.mean(time_inference))  # sample / sec

    experiment.destroy()
