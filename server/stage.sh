#docker pull kaldiasr/kaldi
#docker run -it -v /home/azureuser/forced_final/server:/opt/kaldi/egs/wsj/s5/forced_vit kaldiasr/kaldi /bin/bash
#cd egs/wsj/s5/
#cp -r forced_vit/* .

mkdir model_files
cd model_files
wget http://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz
wget http://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz
wget http://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz
tar -xvzf 0013_librispeech_v1_chain.tar.gz
tar -xvzf 0013_librispeech_v1_extractor.tar.gz
tar -xvzf 0013_librispeech_v1_lm.tar.gz

cd ..
