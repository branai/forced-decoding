// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames, = " << (-weight.Value1() / num_frames)
                << ',' << (weight.Value2() / num_frames);

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier> <GROUND-TRUTH>\n" //FORCED CHANGE: adding argument
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.18;
    bool do_endpointing = false;
    bool online = true;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 6) { //FORCED CHANGE: Adding one for ground truth
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        clat_wspecifier = po.GetArg(5);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_opts.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_opts.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);


    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    CompactLatticeWriter clat_writer(clat_wspecifier);

    OnlineTimingStats timing_stats;

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
      OnlineCmvnState cmvn_state(global_cmvn_stats);

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        feature_pipeline.SetCmvnState(cmvn_state);

        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);

        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);
        OnlineTimer decoding_timer(utt);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          decoding_timer.WaitUntil(samp_offset / samp_freq);
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }

          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
          }

          decoder.AdvanceDecoding();

          if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
            break;
          }
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        //GetDiagnosticsAndPrintOutput(utt, word_syms, clat, &num_frames, &tot_like);









        // BEGIN FORCED DECODING


        std::vector < double > alpha; // forward probability
        std::vector < double > beta; // backward probability

        // From src/latbin/lattice-arc-post.cc
        kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), & clat);
        kaldi::TopSortCompactLatticeIfNeeded( & clat);
        ComputeCompactLatticeAlphas(clat, & alpha);
        ComputeCompactLatticeBetas(clat, & beta);


        // From src/latbin/lattice-arc-post.cc
        double tot_like = beta[clat.Start()];
        int32 state = clat.Start();

        // State info
        bool deadEnd = false; // true if ground truth is not in arks of current state
        int32 best_next_state = -1; // Next best state (in case ground truth is not in ark)
        float32 best_state_prob = 0; // probability of that transition
        //std::vector<int32> state_times;


        // Splitting ground truth
        std::string groundTruth = po.GetArg(6);
        std::vector < std::string > splitGroundTruth;
        SplitStringToVector(groundTruth, " ", true, & splitGroundTruth);


        int32 igt = 0; // ground truth index

        // Output file to store probability sum
        std::ofstream outfile("out.txt");

        if (!outfile.is_open()) {
            std::cerr << "Error opening the file!" << std::endl;
            return 1; // Exit with an error code
        }

        // valid state, deadend, more words left in ground truth
        while (state >= 0 && !deadEnd && igt < splitGroundTruth.size()) {

            // reset state info (to be overwritten)
            deadEnd = true;
            best_state_prob = 0;
            best_next_state = -1;

            for (fst::ArcIterator < CompactLattice > aiter(clat, state); !aiter.Done(); aiter.Next()) {
                const CompactLatticeArc & arc = aiter.Value();
                
                // check if arc matches word in ground truth
                if (word_syms -> Find(splitGroundTruth[igt]) == arc.ilabel) {
                    // loglike probability
                    double arc_loglike = -ConvertToCost(arc.weight) + alpha[state] + beta[arc.nextstate] - tot_like;
                    std::cout << splitGroundTruth[igt]           <<  "         | Probability:" << exp(arc_loglike) << std::endl;
                    
                    // write probablity to output file
                    outfile << exp(arc_loglike) << std::endl;

                    // set next state info and break to go to next state
                    state = arc.nextstate;
                    deadEnd = false;
                    igt++;
                    break;

                } else {

                  // else: curr arc doesnt match
                  // so keep track of the best arc (to use if no matches are found so we can continue decoding)
                  double arc_loglike = -ConvertToCost(arc.weight) + alpha[state] + beta[arc.nextstate] - tot_like;
                  if(exp(arc_loglike) > best_state_prob) {
                    best_state_prob = exp(arc_loglike);
                    best_next_state = arc.nextstate;
                    std::cout << "SETTING BEST STATE " << best_next_state << std::endl;
                    std::cout << "WITH PROB " << best_state_prob << std::endl;
                  }
                }

                /*if (splitGroundTruth[igt] == "<MUS>") {
                    std::cout << std::endl << "***MUSIC STARTING AT FRAME***: " << state_times[state] << std::endl << std::endl;
                    deadEnd = false;
                    igt++;
                    
                    // DFS until we find the next state
                    for (int32 state1 = 0; state1 < clat.NumStates(); ++state1) {
                      for (fst::ArcIterator<CompactLattice> aiter(clat, state1); !aiter.Done(); aiter.Next()) {
                          const CompactLatticeArc &arc = aiter.Value();

                          if(word_syms -> Find(splitGroundTruth[igt]) == arc.ilabel) {
                            state = state1;
                            state1 = clat.NumStates();
                            break;
                          }


                      }
                }
                    break;


                }*/
            }
            // if deadEnd is true, there is a 0 chance the curr word in ground truth was said
            if(deadEnd) {
              outfile << 0 << std::endl;
              std::cout << splitGroundTruth[igt]  <<  "         | Probability:" << 0 << std::endl;

              // set state as the best state found through arc iteration
              state = best_next_state;
              igt++;
              deadEnd = false;
            }
        }
      

        outfile.close();


        // END FORCED DECODING




        




        decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);
        feature_pipeline.GetCmvnState(&cmvn_state);

        // we want to output the lattice with un-scaled acoustics.
        BaseFloat inv_acoustic_scale =
            1.0 / decodable_opts.acoustic_scale;
        ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

        clat_writer.Write(utt, clat);
        KALDI_LOG << "Decoded utterance " << utt;
        num_done++;
      }
    }
    timing_stats.Print(online);

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
