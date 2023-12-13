# forced-decoding
This repository contains<br> 
A) the files needed to perform forced decoding given a piece of audio and 
the "ground truth" to be obtained from the decoding.<br>
B) the files needed to turn this into an API.

Disclaimer: All of the API work here is very primitive and should only be 
used as a proof of concept.

## Requirements
I tested everything here with an Ubuntu 20.04 VM on Azure for the backend 
/ decoding part.

## Setup
### Backend / Forced Decoding
To set up the backend / decoding, follow these steps on the Ubuntu machine 
(as root):
1. Install Docker.
2. `docker pull kaldiasr/kaldi`
3. Clone this directory.
4. `docker run -it -v 
<PATH_TO_REPO>/server:/opt/kaldi/egs/wsj/s5/forced_vit kaldiasr/kaldi 
/bin/bash`
5. In the container, run:<br>
```bash
cd egs/wsj/s5/
cp -r forced_vit/* .
./stage.sh
./make_forced.sh
./setup_speech.sh
```
7. Detach from the container (make sure it stays running)
8. Run the server script `app.py`<br>
`Usage: python3 app.py <docker-container-id>`

### Client Side
No set up needed for client side, just run client script `detect_ans.py` 
with the IP of the server.
`Usage: python3 detect_ans.py <api-ip>`

## Testing
If you just want to test to make sure forced decoding works after setting 
up the backend, inside the docker container you can place the audio file 
here: `/opt/kaldi/egs/wsj/s5/client_sound.wav`

Then run:  `./forced_single.sh "<WORDS-TO-DECODE>"`

Running this for the first time will take a while.

Also, `client/split_test_data.py` is included to show how I roughly split audio given the youtube transcript when testing the decoder for resilience.
