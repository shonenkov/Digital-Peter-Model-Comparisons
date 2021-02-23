# -*- coding: utf-8 -*-
import torch
from tpu_star.experiment import TorchGPUExperiment

from .metrics import cer, wer, string_accuracy


class OCRExperiment(TorchGPUExperiment):

    def calculate_metrics(self, gt_texts, outputs):
        pred_texts = []
        for encoded in outputs.argmax(2).data.cpu().numpy():
            pred_texts.append(self.converter.decode(encoded))
        texts = [self.converter.preprocess(text) for text in gt_texts]
        return {
            'cer': cer(pred_texts, texts),
            'wer': wer(pred_texts, texts),
            'acc': string_accuracy(pred_texts, texts),
        }

    def handle_one_batch(self, batch):
        encoded = batch['encoded'].to(self.device, dtype=torch.int64)
        outputs = self.model(
            batch['image'].to(self.device, dtype=torch.float32),
            encoded[:, :-1], self.is_train,
        )

        if self.config['Prediction'] == 'Attn':
            loss = self.criterion(outputs.transpose(2, 1), encoded[:, 1:])
        elif self.config['Prediction'] == 'CTC':
            lengths = batch['encoded_length'].to(self.device, dtype=torch.int32)
            preds_size = torch.IntTensor([outputs.size(1)] * batch['encoded'].shape[0])
            preds = outputs.log_softmax(2).permute(1, 0, 2)
            loss = self.criterion(preds, encoded, preds_size, lengths)
        else:
            raise

        batch_metrics = self.calculate_metrics(batch['gt_text'], outputs)
        self.metrics.update(loss=loss.detach().cpu().item(), **batch_metrics)

        if self.is_train:
            loss.backward()
            self.optimizer_step()
            self.optimizer.zero_grad()
            self.scheduler.step()
