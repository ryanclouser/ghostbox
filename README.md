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
1. Python 3.x

Usage
---

1. Start GQRX and enable remote control
1. Run the script
   - `python3 ghostbox.py --fm --speed 150 --squelch -30 --bounce`

```
usage: ghostbox.py [-h] [--version] [--ip IP] [--port PORT] [--fm] [--am]
                   [--speed SPEED] [--squelch SQUELCH] [--random] [--forward]
                   [--backward] [--bounce]

Python Ghostbox

options:
  -h, --help         show this help message and exit
  --version          Version
  --ip IP            GQRX IP
  --port PORT        GQRX port
  --fm               Enable FM radio scanning
  --am               Enable AM radio scanning
  --speed SPEED      Scanning speed in milliseconds
  --squelch SQUELCH  Squelch
  --random           Random scanning
  --forward          Forward scanning
  --backward         Backward scanning
  --bounce           Bounce scanning
```

License
---

MIT