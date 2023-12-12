from flask import Flask, request, jsonify
import wave
app = Flask(__name__)

import subprocess


import sys
container_string = ''
if len(sys.argv) > 1:
    container_string = sys.argv[1]
    print("Container String:", container_string)
else:
    print("Usage: python3 app.py <docker-container-id>")

@app.route('/api/getConfidence', methods=['POST'])
def get_confidence():
    try:
        sound_data_array = request.json['sound']
        sound_data_bytes = bytes(sound_data_array)

        with wave.open('client_sound.wav', 'wb') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(request.json['fr'])
            wave_file.writeframes(sound_data_bytes)

        ground_truth = request.json['ground_truth']

        cmd_dbg_make = f'docker exec -it -w /opt/kaldi/egs/wsj/s5 {container_string} /bin/bash /opt/kaldi/egs/wsj/s5/make_forced.sh'
        res_dbg_make = subprocess.run(cmd_dbg_make, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res_dbg_make.stdout)


        cmd_run = f'docker exec -it -w /opt/kaldi/egs/wsj/s5 {container_string} /bin/bash /opt/kaldi/egs/wsj/s5/forced_single.sh "{ground_truth}"'
        res_run = subprocess.run(cmd_run, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res_run.stdout)

        cmd_get_output = f'docker exec -it -w /opt/kaldi/egs/wsj/s5 {container_string} cat /opt/kaldi/egs/wsj/s5/out.txt'
        res_get_output = subprocess.run(cmd_get_output, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res_get_output.stdout)


        return str(res_get_output.stdout), 200
    except Exception as e:
        print('Error:', str(e))
        return 'Server Error', 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)