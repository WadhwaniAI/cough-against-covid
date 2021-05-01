### Defaults
Configs in this folder can be used to add a new functionality or as a starting point for your custom model. For instance if we add label-smoothing loss, there would be a `label-smoothing-random.yml`. 

**[1d.yml](1d.yml)** : Uses 1D CNNs on raw waveforms.

**[classical-baseline.yml](classical-baseline.yml)** : Uses a classical model on MFCC features to perform covid detection.

**[context-neural.yml](context-neural.yml)** : Uses a deep neural model for contextual data to perform covid detection.

**[file-agg.yml](file-agg.yml)** : Uses chunk-aggregation for validation set.

**[label-smoothing-random.yml](label-smoothing-random.yml)** : Uses label smoothing loss for covid detection task on wiai data.

**[multi-signal-training.yml](multi-signal-training.yml)** : Uses a deep model to perform multi-signal covid detection using cough and context.

**[stft.yml](stft.yml)** : Uses simple STFT transform on a raw audio waveform.

**[unsupervised.yml](unsupervised.yml)** : Uses handcrafted features (AI4COVID/Cambridge) to visualize.

**[wiai.yml](wiai.yml)** : Uses simple spectrograms on WIAI datasets.

**[with-frames.yml](with-frames.yml)** : Uses simple spectrograms on WIAI datasets with inputs split in frames of 2 seconds.
