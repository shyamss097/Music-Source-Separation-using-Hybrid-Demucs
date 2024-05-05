#Music Source Separation using Hybrid Demucs


### Running from Colab

I made a Colab to easily separate track with Demucs. Note that
transfer speeds with Colab are a bit slow for large media files,
but it will allow you to use Demucs without installing anything.

[Demucs on Google Colab](https://colab.research.google.com/drive/1dC9nVxk3V_VPjUADsnFu8EiT-xnU1tGH?usp=sharing)

### Web Demo

Integrated to [Hugging Face Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/demucs)

### Graphical Interface

@CarlGao4 has released a GUI for Demucs: [CarlGao4/Demucs-Gui](https://github.com/CarlGao4/Demucs-Gui). Downloads for Windows and macOS is available [here](https://github.com/CarlGao4/Demucs-Gui/releases). Use [FossHub mirror](https://fosshub.com/Demucs-GUI.html) to speed up your download.

@Anjok07 is providing a self contained GUI in [UVR (Ultimate Vocal Remover)](https://github.com/facebookresearch/demucs/issues/334) that supports Demucs.

### Other providers

Audiostrip is providing free online separation with Demucs on their website [https://audiostrip.co.uk/](https://audiostrip.co.uk/).

[MVSep](https://mvsep.com/) also provides free online separation, select `Demucs3 model B` for the best quality.

[Neutone](https://neutone.space/) provides a realtime Demucs model in their free VST/AU plugin that can be used in your favorite DAW.


## Separating tracks

In order to try Demucs, you can just run from any folder (as long as you properly installed it)

```bash
demucs PATH_TO_AUDIO_FILE_1 [PATH_TO_AUDIO_FILE_2 ...]   # for Demucs
# If you used `pip install --user` you might need to replace demucs with python3 -m demucs
python3 -m demucs --mp3 --mp3-bitrate BITRATE PATH_TO_AUDIO_FILE_1  # output files saved as MP3
        # use --mp3-preset to change encoder preset, 2 for best quality, 7 for fastest
# If your filename contain spaces don't forget to quote it !!!
demucs "my music/my favorite track.mp3"
# You can select different models with `-n` mdx_q is the quantized model, smaller but maybe a bit less accurate.
demucs -n mdx_q myfile.mp3
# If you only want to separate vocals out of an audio, use `--two-stems=vocals` (You can also set to drums or bass)
demucs --two-stems=vocals myfile.mp3
```


If you have a GPU, but you run out of memory, please use `--segment SEGMENT` to reduce length of each split. `SEGMENT` should be changed to a integer describing the length of each segment in seconds.
A segment length of at least 10 is recommended (the bigger the number is, the more memory is required, but quality may increase). Note that the Hybrid Transformer models only support a maximum segment length of 7.8 seconds.
Creating an environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is also helpful. If this still does not help, please add `-d cpu` to the command line. See the section hereafter for more details on the memory requirements for GPU acceleration.

Separated tracks are stored in the `separated/MODEL_NAME/TRACK_NAME` folder. There you will find four stereo wav files sampled at 44.1 kHz: `drums.wav`, `bass.wav`,
`other.wav`, `vocals.wav` (or `.mp3` if you used the `--mp3` option).

All audio formats supported by `torchaudio` can be processed (i.e. wav, mp3, flac, ogg/vorbis on Linux/macOS, etc.). On Windows, `torchaudio` has limited support, so we rely on `ffmpeg`, which should support pretty much anything.
Audio is resampled on the fly if necessary.
The output will be a wav file encoded as int16.
You can save as float32 wav files with `--float32`, or 24 bits integer wav with `--int24`.
You can pass `--mp3` to save as mp3 instead, and set the bitrate (in kbps) with `--mp3-bitrate` (default is 320).

It can happen that the output would need clipping, in particular due to some separation artifacts.
Demucs will automatically rescale each output stem so as to avoid clipping. This can however break
the relative volume between stems. If instead you prefer hard clipping, pass `--clip-mode clamp`.
You can also try to reduce the volume of the input mixture before feeding it to Demucs.


Other pre-trained models can be selected with the `-n` flag.
The list of pre-trained models is:
- `htdemucs`: first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
- `htdemucs_ft`: fine-tuned version of `htdemucs`, separation will take 4 times more time
    but might be a bit better. Same training set as `htdemucs`.
- `htdemucs_6s`: 6 sources version of `htdemucs`, with `piano` and `guitar` being added as sources.
    Note that the `piano` source is not working great at the moment.
- `hdemucs_mmi`: Hybrid Demucs v3, retrained on MusDB + 800 songs.
- `mdx`: trained only on MusDB HQ, winning model on track A at the [MDX][mdx] challenge.
- `mdx_extra`: trained with extra training data (**including MusDB test set**), ranked 2nd on the track B
    of the [MDX][mdx] challenge.
- `mdx_q`, `mdx_extra_q`: quantized version of the previous models. Smaller download and storage
    but quality can be slightly worse.
- `SIG`: where `SIG` is a single model from the [model zoo](docs/training.md#model-zoo).

The `--two-stems=vocals` option allows separating vocals from the rest of the accompaniment (i.e., karaoke mode).
`vocals` can be changed to any source in the selected model.
This will mix the files after separating the mix fully, so this won't be faster or use less memory.

The `--shifts=SHIFTS` performs multiple predictions with random shifts (a.k.a the *shift trick*) of the input and average them. This makes prediction `SHIFTS` times
slower. Don't use it unless you have a GPU.

The `--overlap` option controls the amount of overlap between prediction windows. Default is 0.25 (i.e. 25%) which is probably fine.
It can probably be reduced to 0.1 to improve a bit speed.


The `-j` flag allow to specify a number of parallel jobs (e.g. `demucs -j 2 myfile.mp3`).
This will multiply by the same amount the RAM used so be careful!

### Memory requirements for GPU acceleration

If you want to use GPU acceleration, you will need at least 3GB of RAM on your GPU for `demucs`. However, about 7GB of RAM will be required if you use the default arguments. Add `--segment SEGMENT` to change size of each split. If you only have 3GB memory, set SEGMENT to 8 (though quality may be worse if this argument is too small). Creating an environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` can help users with even smaller RAM such as 2GB (I separated a track that is 4 minutes but only 1.5GB is used), but this would make the separation slower.

If you do not have enough memory on your GPU, simply add `-d cpu` to the command line to use the CPU. With Demucs, processing time should be roughly equal to 1.5 times the duration of the track.

## Calling from another Python program

The main function provides an `opt` parameter as a simple API. You can just pass the parsed command line as this parameter: 
```python
# Assume that your command is `demucs --mp3 --two-stems vocals -n mdx_extra "track with space.mp3"`
# The following codes are same as the command above:
import demucs.separate
demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "track with space.mp3"])

# Or like this
import demucs.separate
import shlex
demucs.separate.main(shlex.split('--mp3 --two-stems vocals -n mdx_extra "track with space.mp3"'))
```

To use more complicated APIs, see [API docs](docs/api.md)

## Training Demucs

If you want to train (Hybrid) Demucs, please follow the [training doc](docs/training.md).

## MDX Challenge reproduction

In order to reproduce the results from the Track A and Track B submissions, checkout the [MDX Hybrid Demucs submission repo][mdx_submission].



## How to cite

```
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

## License

Demucs is released under the MIT license as found in the [LICENSE](LICENSE) file.

[hybrid_paper]: https://arxiv.org/abs/2111.03600
[waveunet]: https://github.com/f90/Wave-U-Net
[musdb]: https://sigsep.github.io/datasets/musdb.html
[openunmix]: https://github.com/sigsep/open-unmix-pytorch
[mmdenselstm]: https://arxiv.org/abs/1805.02410
[demucs_v2]: https://github.com/facebookresearch/demucs/tree/v2
[demucs_v3]: https://github.com/facebookresearch/demucs/tree/v3
[spleeter]: https://github.com/deezer/spleeter
[soundcloud]: https://soundcloud.com/honualx/sets/source-separation-in-the-waveform-domain
[d3net]: https://arxiv.org/abs/2010.01733
[mdx]: https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
[kuielab]: https://github.com/kuielab/mdx-net-submission
[decouple]: https://arxiv.org/abs/2109.05418
[mdx_submission]: https://github.com/adefossez/mdx21_demucs
[bandsplit]: https://arxiv.org/abs/2209.15174
[htdemucs]: https://arxiv.org/abs/2211.08553
