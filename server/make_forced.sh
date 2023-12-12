cd /opt/kaldi/egs/wsj/s5/
cp forced_vit/online2-wav-nnet3-latgen-faster.cc /opt/kaldi/src/online2bin/ 
cd /opt/kaldi/src/online2bin/
make
cd /opt/kaldi/egs/wsj/s5/