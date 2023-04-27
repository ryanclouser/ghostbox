Python Ghostbox
===

```
   _____ _               _   _               
  / ____| |             | | | |              
 | |  __| |__   ___  ___| |_| |__   _____  __
 | | |_ | '_ \ / _ \/ __| __| '_ \ / _ \ \/ /
 | |__| | | | | (_) \__ \ |_| |_) | (_) >  < 
  \_____|_| |_|\___/|___/\__|_.__/ \___/_/\_\
                                             
```

Scans through AM/FM radio stations using SDR and Python.

Requirements
---

1. RTL-SDR or equivalent
1. GQRX
1. Python 3.4+

**STT / TTS**

1. torch
1. torchaudio
1. sounddevice
1. numpy
1. omegaconf
1. sentencepiece
1. soundfile

Usage
---

1. `pip3 install -r requirements.txt`
1. Start GQRX and enable remote control
1. Start UDP for speech to text if needed (reconfigure the UDP host to `127.0.0.1` in GQRX)
1. Run the script
   - `python3 ghostbox.py --fm -i 0.15 --bounce`

```
usage: ghostbox.py [-h] [-v] [--ip IP] [-p PORT] [--fm] [--am] [-i INTERVAL] [-s SQUELCH] [--random] [--forward]
                   [--backward] [--bounce] [--stt] [-w] [--tts] [-r] [-l]

Ghostbox

options:
  -h, --help            show this help message and exit
  -v, --version         display the program version
  --ip IP               gqrx IP
  -p PORT, --port PORT  gqrx port
  --fm                  enable FM radio scanning
  --am                  enable AM radio scanning
  -i INTERVAL, --interval INTERVAL
                        scanning interval in seconds
  -s SQUELCH, --squelch SQUELCH
                        squelch
  --random              random scanning
  --forward             forward scanning
  --backward            backward scanning
  --bounce              bounce scanning
  --stt                 enable speech to text
  -w, --wordlist        use a wordlist after audio is processed
  --tts                 enable text to speech
  -r, --reverb          apply reverb effect to TTS
  -l, --long-words      hide short words from the output
```

Speech to Text
---

The Ghostbox is capable of speech recognition by processing the UDP audio stream from GQRX. GQRX can be muted and audio will still be processed.

```
python3 ghostbox.py --fm -i 0.15 --bounce --stt -w --tts
```

macOS
---

1. Install [homebrew](https://brew.sh/)
1. Install miniconda
   - `brew install miniconda`
   - `conda init "$(basename "${SHELL}")"`
1. Restart your shell
1. Install PyTorch
   - `conda install pytorch torchvision torchaudio -c pytorch`
1. Install Python requirements
   - `pip3 install -r requirements.txt`

**Build GQRX for M1**

If you get an error about libxml2, just ignore it and copy the `GQRX.app` file to your Applications folder.

- https://gist.github.com/Forst/db68106136be3380086e3c38be094d99

Wordlist
---

Top 5000 common English words.

- https://github.com/filiph/english_words/

Errors
---

**UDP Bind Error**

It's possible a copy of the Python script is still running in the background.

```
kill -9 $(ps -A | grep python | awk '{print $1}')
```

License
---

MIT