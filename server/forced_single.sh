ground_truth="$1"
wav_file="client_sound.wav"

ffmpeg -y -i forced_vit/$wav_file -ar 16000 /opt/kaldi/egs/wsj/s5/$wav_file

echo "utt1 /opt/kaldi/egs/wsj/s5/$wav_file" > wav.scp
echo "utt1 spk1" > utt2spk

utils/utt2spk_to_spk2utt.pl ./utt2spk > ./high_res/spk2utt
cp wav.scp ./high_res/wav.scp
cp utt2spk ./high_res/utt2spk


./path.sh
/opt/kaldi/egs/wsj/s5/steps/make_mfcc.sh   --nj 1   --mfcc-config /opt/kaldi/egs/wsj/s5/conf/mfcc_hires.conf   ./high_res
steps/online/nnet2/extract_ivectors_online.sh --nj 1 high_res/ model_files/exp/nnet3_cleaned/extractor model_files/exp/nnet3_cleaned/ivectors_high_res
utils/mkgraph.sh --self-loop-scale 1.0 model_files/data/lang_test_tgsmall model_files/exp/chain_cleaned/tdnn_1d_sp model_files/exp/chain_cleaned/tdnn_1d_sp/graph_tgsmall



/opt/kaldi/src/online2bin/online2-wav-nnet3-latgen-faster     \
    --do-endpointing=false \
    --frames-per-chunk=20 \
    --extra-left-context-initial=0 \
    --frame-subsampling-factor=3 \
    --config=/opt/kaldi/egs/wsj/s5/conf/online_cmvn.conf \
    --mfcc-config=/opt/kaldi/egs/wsj/s5/conf/mfcc_hires.conf \
    --min-active=0 \
    --max-active=6000 \
    --beam=11 \
    --lattice-beam=6.0 \
    --acoustic-scale=1.0 \
    --word-symbol-table=/opt/kaldi/egs/wsj/s5/model_files/exp/chain_cleaned/tdnn_1d_sp/graph_tgsmall/words.txt \
    --online=true \
    --ivector-extraction-config=model_files/exp/nnet3_cleaned/ivectors_high_res/conf/ivector_extractor.conf /opt/kaldi/egs/wsj/s5/model_files/exp/chain_cleaned/tdnn_1d_sp/final.mdl /opt/kaldi/egs/wsj/s5/model_files/exp/chain_cleaned/tdnn_1d_sp/graph_tgsmall/HCLG.fst ark:$PWD/high_res/spk2utt scp:$PWD/wav.scp ark,scp:lattice.ark,lattice.scp "$ground_truth"