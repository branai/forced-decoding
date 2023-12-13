wav_file="client_sound.wav"

echo "utt1 /opt/kaldi/egs/wsj/s5/$wav_file" > wav.scp
echo "utt1 spk1" > utt2spk

utils/utt2spk_to_spk2utt.pl ./utt2spk > ./high_res/spk2utt
cp wav.scp ./high_res/wav.scp
cp utt2spk ./high_res/utt2spk


./path.sh
/opt/kaldi/egs/wsj/s5/steps/make_mfcc.sh   --nj 1   --mfcc-config /opt/kaldi/egs/wsj/s5/conf/mfcc_hires.conf   ./high_res
steps/online/nnet2/extract_ivectors_online.sh --nj 1 high_res/ model_files/exp/nnet3_cleaned/extractor model_files/exp/nnet3_cleaned/ivectors_high_res
utils/mkgraph.sh --self-loop-scale 1.0 model_files/data/lang_test_tgsmall model_files/exp/chain_cleaned/tdnn_1d_sp model_files/exp/chain_cleaned/tdnn_1d_sp/graph_tgsmall