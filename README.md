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

STT / TTS
---

1. torch
1. torchaudio
1. sounddevice

Usage
---

1. `pip3 install -r requirements.txt`
1. Start GQRX and enable remote control
1. Run the script
   - `python3 ghostbox.py --fm --speed 150 --squelch -30 --bounce`

```
usage: ghostbox.py [-h] [--version] [--ip IP] [--port PORT] [--fm] [--am] [--speed SPEED] [--squelch SQUELCH] [--random] [--forward] [--backward] [--bounce] [--record] [--record-interval RECORD_INTERVAL] [--wordlist] [--tts]

Python Ghostbox

options:
  -h, --help            show this help message and exit
  --version             Version
  --ip IP               GQRX IP
  --port PORT           GQRX port
  --fm                  Enable FM radio scanning
  --am                  Enable AM radio scanning
  --speed SPEED         Scanning speed in milliseconds
  --squelch SQUELCH     Squelch
  --random              Random scanning
  --forward             Forward scanning
  --backward            Backward scanning
  --bounce              Bounce scanning
  --record              Record and process audio
  --record-interval RECORD_INTERVAL
                        Time for each audio recording
  --wordlist            Use a wordlist after audio is processed
  --tts                 Use text to speech
```

Recording
---

The Ghostbox is capable of speech recognition by processing the output WAV file from GQRX. The default behavior is to record for 5s, stop recording, process the WAV, delete the file, then start recording again. GQRX can be muted and recordings will function normally. Any pull requests to stream the WAV directly into the ASR would be more than welcome :)

```
python3 ghostbox.py --fm --speed 150 --squelch -30 --bounce --record --wordlist --tts
```

Wordlist
---

Top 5000 common English words.

- https://github.com/filiph/english_words/

License
---

MIT