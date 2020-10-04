
import sys
sys.path.append('../../delve')

import delve


import pandas as pd

import traja






def test_from_df():
    df = traja.generate(n=20)

    df = df.filter(items=['x', 'y'])

    save_path = 'temp/test'

    warmup_steps = 1

    # Run 1
    timeseries_method = 'last_timestep'

    model = traja.models.LSTM(input_size=2, hidden_size=2, num_layers=3, output_size=2, dropout=0, bidirectional=False)
    writer = delve.writers.CSVandPlottingWriter(save_path, fontsize=16, primary_metric='test_accuracy')
    saturation = delve.CheckLayerSat(save_path,
                                     [writer], model,
                                     stats=['embed'],
                                     timeseries_method=timeseries_method)

    sequence_length = 5
    train_fraction = .25
    batch_size = 1
    shift = 2

    train_loader, test_loader = traja.models.get_timeseries_data_loaders(df, sequence_length,
                                                                         train_fraction, batch_size, shift)

    trainer = traja.models.Trainer(model, train_loader, test_loader, epochs=6, optimizer="adam",
                                   warmup_steps=warmup_steps)

    trainer.train()

    pass