Technical Design Document: Automated Indian-English CEFR Speech Assessment Platform1. System Overview & Problem StatementStandard English pronunciation assessment systems (such as Whisper or LibriSpeech-trained Wav2Vec models) evaluate acoustic speech against standard American (GA) or British (RP) phonetic baselines. When deployed on Indian English speakers, these models introduce systematic bias:Phonetic False Negatives: Retroflex stops ($[ʈ], [ɖ]$) are penalized as distorted alveolar stops ($/t/, /d/$). Dental stops ($[t̪], [d̪]$) are penalized as mispronounced dental fricatives ($/θ/, /ð/$ in "think" and "this"). The labiodental approximant ($[\upsilon]$) is flagged as a confusion between $/v/$ and $/w/$.Prosodic Misinterpretations: Syllable-timed cadence is misidentified as poor fluency or broken pausing when compared to stress-timed expectations.Lexical Normalization: Legitimate Indian English idioms and lexical terms (prepone, batchmate, pass out) are frequently misheard or corrected to Western nearest-matches.Architectural Limitations of Audio-LLMs: Generative Audio-LLMs pool acoustic frames across 40–80 ms windows, destroying sub-phonemic acoustic boundaries and preventing millisecond-level visual feedback (Green/Yellow/Red). Furthermore, unconstrained autoregressive sampling produces fluctuating, non-deterministic scores across identical inputs.Solution Pattern: The Decoupled Multimodal ArchitectureThis engine decouples transcription from acoustic scoring, processing speech through four distinct stages:Linguistic ASR: Transcribes accented speech verbatim without acoustic normalization.Acoustic Phonetics: Evaluates raw articulation and Goodness of Pronunciation (GOP) using native Indian acoustic representations.Semantic & Syntactic Profiling: Extracts discourse structure, syntactic depth, and task relevance.Multimodal Fusion Regressor: An ensemble tree regressor calibrated via Item Response Theory (IRT) that maps the feature space to continuous CEFR scores ($1.00$ to $6.00$).2. End-to-End System Architecture                                      [User Audio: 16 kHz Mono PCM]
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
    [Model 1: Linguistic ASR]                                             [Model 2: Acoustic Phonetics]
  ai4bharat/indic-conformer-600m-multilingual                            ai4bharat/indicwav2vec-base
  - RNN-T Decoder (English head)                                         - Fine-tuned with LoRA on Svarah
  - Verbatim Indian English transcript                                   - Frame posteriors P(phone|t) at 50 Hz
                 │                                                                     │
                 ▼                                                                     ▼
   [Punctuation & Truecasing]                                            [Allophone-Aware GOP Engine]
   deepmultilingualpunctuation                                           - Candidate set relaxation: A(p)
                 │                                                       - Log-likelihood ratio calculation
                 ▼                                                       - Temporal pause & fluency metrics
       Punctuated Transcript                                                           │
                 │                                                                     │
       ┌─────────┴─────────┐                                                           │
       ▼                   ▼                                                           │
 [spaCy Syntactic]   [bge-small Semantic]                                              │
 - Tree depth        - Task relevance (cosine)                                         │
 - Clause density    - Adjacent coherence                                              │
 - CEFR-J Lexicon    - Masked LM perplexity                                            │
       │                   │                                                           │
       └─────────┬─────────┘                                                           │
                 ▼                                                                     ▼
      [Linguistic Features]                                                  [Acoustic Features]
      (Cols 10–21)                                                           (Cols 0–9)
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                    [22-Dimensional Feature Vector]
                                                    │
                                                    ▼
                                      [XGBoost Regression Head]
                                    (Trained on MFRM Target Data)
                                                    │
                                                    ▼
                                 [Continuous CEFR Score: 1.00 – 6.00]
                                 ├── Discrete Band: A1, A2, B1, B2, C1, C2
                                 └── Phonetic Diagnostics: Green / Yellow / Red
3. Mathematical Specifications & Core AlgorithmsAlgorithm 1: Allophone-Relaxed Goodness of Pronunciation (GOP)Standard Goodness of Pronunciation calculates the posterior probability of a canonical phoneme $p$ relative to all competing acoustic paths across the aligned frame interval $T_p = [t_s, t_e]$:$$\text{GOP}_{\text{Standard}}(p) = \frac{1}{\vert{}T_p\vert{}} \sum_{t=t_s}^{t_e} \left[ \log P(p \mid x_t) - \max_{q \neq p} \log P(q \mid x_t) \right]$$To eliminate accent-induced false negatives, the acoustic target is expanded to an Indian English Allophonic Equivalence Class $\mathcal{A}(p)$. The model extracts the maximum posterior likelihood across all valid regional variants and contrasts it strictly against non-allophonic phonemes:$$\text{GOP}_{\text{Indian}}(p) = \frac{1}{\vert{}T_p\vert{}} \sum_{t=t_s}^{t_e} \left[ \max_{a \in \mathcal{A}(p)} \log P(a \mid x_t) - \max_{q \notin \mathcal{A}(p)} \log P(q \mid x_t) \right]$$Allophonic Equivalence Dictionary $\mathcal{A}(p)$Canonical Phone (p)   Standard IPA     Accepted Allophone Set A(p)       Target Articulation Shift
──────────────────────────────────────────────────────────────────────────────────────────────────────────
/t/                   [t]              { T, T_RETROFLEX, D }             Alveolar stop -> Retroflex stop [ʈ]
/d/                   [d]              { D, D_RETROFLEX, T }             Alveolar stop -> Retroflex stop [ɖ]
/θ/                   [θ]              { TH, T_DENTAL, T }               Dental fricative -> Dental stop [t̪]
/ð/                   [ð]              { DH, D_DENTAL, D }               Voiced fricative -> Dental stop [d̪]
/v/, /w/              [v], [w]         { V, W }                          Labiodental approximant [ʋ]
/r/                   [ɹ]              { R, R_FLAP }                     Post-alveolar -> Alveolar flap [ɾ]
/iː/, /ɪ/             [iː], [ɪ]        { IY, IH }                        Neutralization of tense/lax length
/uː/, /ʊ/             [uː], /ʊ/        { UW, UH }                        Neutralization of tense/lax length
Diagnostic Threshold MappingRaw log-likelihood ratios are mapped to a bounded score interval $[0, 100]$ via a calibrated logistic function:$$\text{Score}(p) = \frac{100}{1 + \exp\left(-\gamma \cdot \text{GOP}_{\text{Indian}}(p)\right)}, \quad \text{where } \gamma = 1.8$$The resulting score drives the UI color states:$\text{Score}(p) \ge 75.0$: GREEN (Articulated accurately or matches standard Indian English allophone).$50.0 \le \text{Score}(p) < 75.0$: YELLOW (Acoustically blurred, muffled, or inconsistent duration).$\text{Score}(p) < 50.0$: RED (Phonemic substitution, omission, or major distortion).Algorithm 2: Multi-Faceted Rasch Measurement (MFRM) for Ground-Truth CalibrationCalibration datasets such as speechocean762 provide ratings across multiple human examiners. Using raw arithmetic means preserves individual examiner severity bias (strict vs. lenient raters) and prompt-specific difficulty bias.The Multi-Faceted Rasch Measurement (MFRM) model decomposes each observed rating $S_{nijk}$ into additive, separable latent components:$$S_{nijk} = \theta_n - \beta_i - \delta_j + \epsilon_{nijk}$$$\theta_n$: True latent speaking proficiency of candidate $n$ (the de-biased target parameter).$\beta_i$: Inherent acoustic/phonetic difficulty of prompt/sentence $i$.$\delta_j$: Severity offset parameter of human examiner $j$.$\epsilon_{nijk}$: Residual error term, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.Ordinary Least Squares Fixed-Effects EstimationThe parameters are solved across the rating matrix via Two-Way Fixed Effects:$$\mathbf{S} = \mathbf{X}_\theta \boldsymbol{\theta} + \mathbf{X}_\beta \boldsymbol{\beta} + \mathbf{X}_\delta \boldsymbol{\delta} + \boldsymbol{\epsilon}$$Once the vector of candidate latent parameters $\boldsymbol{\theta} = (\theta_1, \dots, \theta_N)$ is isolated, it is standardized and mapped to the continuous CEFR continuum $[1.00, 6.00]$:$$\mu_\theta = \frac{1}{N} \sum_{n=1}^N \theta_n, \qquad \sigma_\theta = \sqrt{\frac{1}{N} \sum_{n=1}^N (\theta_n - \mu_\theta)^2}$$$$z_n = \frac{\theta_n - \mu_\theta}{\sigma_\theta}$$$$y_n = \text{clip}\left( 3.20 + (z_n \times 0.95), \quad 1.00, \quad 6.00 \right)$$Midpoint Anchor ($3.20$): Centers the demographic distribution at B1.Spread Scaling ($0.95$): Scales the standard distribution such that $\pm 2\sigma$ spans from A1 (1.00) to C2 (6.00).Algorithm 3: Semantic Coherence & Task Relevance FormulationThe platform evaluates semantic discourse and topical alignment using dense embeddings:Task Relevance ($\mathcal{R}_{\text{task}}$): Quantifies alignment between the candidate's transcript and the evaluation prompt:$$\mathcal{R}_{\text{task}} = \frac{\mathbf{e}_{\text{prompt}} \cdot \mathbf{e}_{\text{transcript}}}{\Vert{}\mathbf{e}_{\text{prompt}}\Vert{} \Vert{}\mathbf{e}_{\text{transcript}}\Vert{}}$$Where $\mathbf{e} \in \mathbb{R}^{384}$ is extracted using BAAI/bge-small-en-v1.5.Inter-Sentence Local Coherence ($\mu_{\text{coh}}, \sigma_{\text{coh}}$): Measures discourse progression across sequential sentences $(s_1, s_2, \dots, s_M)$:$$\text{sim}_m = \cos(\mathbf{e}_{s_m}, \mathbf{e}_{s_{m+1}})$$$$\mu_{\text{coh}} = \frac{1}{M-1} \sum_{m=1}^{M-1} \text{sim}_m, \qquad \sigma_{\text{coh}} = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M-1} (\text{sim}_m - \mu_{\text{coh}})^2}$$Masked Language Model Perplexity ($\text{PPL}_{\text{pseudo}}$): Evaluates natural collocations and penalizes unnatural phrasing using distilroberta-base. Tokens are masked sequentially with probability $p=0.15$:$$\text{PPL}_{\text{pseudo}} = \exp\left( -\frac{1}{\vert{}K\vert{}} \sum_{k \in K} \log P(w_k \mid w_{\setminus k}) \right)$$4. The 22-Dimensional Feature Vector SpecificationThe table below defines the full schema ingested by the XGBoost regression head:IndexFeature IdentifierMathematical Definition / Extraction SourceRange[0]mean_gopOverall average of $\text{GOP}_{\text{Indian}}(p)$ across all target phonemes$[0.0, 100.0]$[1]vowel_gop_meanAverage GOP isolated to vowel phonemes $\{AA, AE, AH, \dots\}$$[0.0, 100.0]$[2]consonant_gop_meanAverage GOP isolated to consonant phonemes$[0.0, 100.0]$[3]low_gop_ratio$\frac{\text{Count}(\text{Score}(p) < 50.0)}{\text{Total Phonemes}}$$[0.0, 1.0]$[4]indian_allophone_rt$\frac{\text{Count}(p \in \mathcal{A}(p) \setminus \{p\})}{\text{Total Consonants}}$$[0.0, 1.0]$[5]speech_rate$\frac{\text{Total Syllables}}{T_{\text{total}}}$ (syllables per second)$[0.0, 8.0]$[6]articulation_rate$\frac{\text{Total Syllables}}{T_{\text{active}}}$ (excluding pauses $>250\text{ ms}$)$[0.0, 10.0]$[7]pause_to_speech_rt$\frac{T_{\text{pause}}}{T_{\text{total}}}$ (pause frames $>250\text{ ms}$)$[0.0, 1.0]$[8]mean_run_lengthAverage number of uninterrupted syllables between pauses$[1.0, 25.0]$[9]f0_pitch_varianceStandard deviation of fundamental frequency contour ($\sigma_{F_0}$)$[0.0, 60.0]$[10]max_tree_depth$\max_{s \in S} (\text{depth}(\text{parse\_tree}(s)))$ via spaCy$[1, 20]$[11]mean_tree_depthMean dependency tree depth across all sentences$[1.0, 15.0]$[12]clause_density$\frac{\text{Count}(advcl + relcl + ccomp + xcomp)}{\text{Total Sentences}}$$[0.0, 6.0]$[13]mean_sent_length$\frac{\text{Total Words}}{\text{Total Sentences}}$$[1.0, 50.0]$[14]pct_a1_a2Proportion of words matched to CEFR A1–A2 lemma lists$[0.0, 1.0]$[15]pct_b1_b2Proportion of words matched to CEFR B1–B2 lemma lists$[0.0, 1.0]$[16]pct_c1_c2Proportion of words matched to CEFR C1–C2 lemma lists$[0.0, 1.0]$[17]mtld_diversityMeasure of Textual Lexical Diversity (factor-retention)$[10.0, 120.0]$[18]task_relevance$\cos(\mathbf{e}_{\text{prompt}}, \mathbf{e}_{\text{transcript}})$ via bge-small$[-1.0, 1.0]$[19]local_coherence_muMean cosine similarity between consecutive sentences$[-1.0, 1.0]$[20]local_coherence_sdStandard deviation of cosine similarities across sentences$[0.0, 1.0]$[21]masked_perplexityPseudo-perplexity computed over masked sequence via DistilRoBERTa$[1.0, 100.0]$5. Model Configuration & Training BlueprintModel 1: Linguistic ASR (IndicConformer-600M Multilingual)Checkpoint: ai4bharat/indic-conformer-600m-multilingualArchitecture: Hybrid Conformer (24 FastConformer encoder layers, 17 depthwise-separable convolutional blocks, RNN-T and CTC decoders).Execution: FP16 on GPU (language_id="en", decoder="rnnt").VRAM Footprint: ~2.2 GB.Role: Freezes acoustic representations and produces unpunctuated verbatim transcripts.Model 2: Acoustic Phonetic Scorer (IndicWav2Vec-Base + LoRA)Checkpoint: ai4bharat/indicwav2vec-hindi (or indicwav2vec_v1_base).Architecture: 7-layer CNN temporal feature extractor + 12 Transformer encoder blocks ($d_{\text{model}}=768$).Adaptation:Freeze the 7-layer CNN feature extractor entirely.Freeze Transformer blocks 0 through 8.Attach LoRA ($r=16, \alpha=32$, dropout $0.05$) to query, key, value, and output projection layers of blocks 9–11.Replace the original output head with a linear classification layer: nn.Linear(768, 44) (mapping to ARPABET phonemes).Training Data: ai4bharat/Svarah (117 speakers, 19 Indian states, CC-BY 4.0).Supervision: Canonical transcripts converted via G2P with Indian allophonic target expansion and supervised under CTC loss:$$\mathcal{L}_{\text{CTC}} = -\log P(\mathbf{p}_{\text{target}} \mid \mathbf{x})$$Model 3: Punctuation & NLP Parsing StackPunctuation & Truecasing: oliverguhr/fullstop-punctuation-multilingual-sonar-base (runs in FP32 on CPU, ~280 MB RAM). Restores sentence boundaries required for dependency trees.Syntactic Parser: spaCy en_core_web_sm (runs on CPU). Extracts dependency tree depths and clause counts.Semantic Embedder: BAAI/bge-small-en-v1.5 (runs on GPU, ~0.25 GB VRAM). Generates normalized 384-dimensional sentence vectors.Perplexity Estimator: distilroberta-base (runs on GPU, ~0.3 GB VRAM). Computes pseudo-perplexity across masked inputs.XGBoost Fusion RegressorObjective: reg:squarederrorHyperparameters:n_estimators: 300max_depth: 4learning_rate: 0.03subsample: 0.85colsample_bytree: 0.85Target ($y$): MFRM-calibrated continuous CEFR rating ($1.00$ to $6.00$).6. Score Mapping & Diagnostic ReportingContinuous Score to CEFR Band AssignmentThe XGBoost regressor outputs a continuous prediction $y \in [1.00, 6.00]$. The score maps directly into official CEFR bands:Score Range (y)    CEFR Band               Acoustic & Linguistic Diagnostic Profile
──────────────────────────────────────────────────────────────────────────────────────────────────────────
[1.00, 1.99]       A1: Breakthrough        Fragmented speech, mean GOP < 50%, pause ratio > 40%.
                                           Short isolated words, syntactic depth <= 2, pct_a1 > 80%.

[2.00, 2.99]       A2: Waystage            Basic sentence coordination (and, but), slow articulation.
                                           High retroflex presence, low vowel stability, limited vocab.

[3.00, 3.99]       B1: Threshold           Connected discourse, basic causal links (because, although).
                                           Stable Indian-English cadence, acceptable consonant GOP.

[4.00, 4.99]       B2: Vantage             Complex clauses present, subordinate branching, C1 vocab > 10%.
                                           Clear phoneme articulation, expressive intonation contour.

[5.00, 5.99]       C1: Effective           Fluent connected speech, native-like contrastive stress.
                   Operational             Complex embedded clauses, low perplexity, coherence > 0.75.

6.00               C2: Mastery             Effortless, natural articulation with full idiomatic precision.
                                           Extensive vocabulary, structurally integrated discourse.
Complete System Output PayloadJSON{
  "assessment_metadata": {
    "sample_id": "eval_ind_00941",
    "duration_seconds": 6.84,
    "target_prompt": "Explain the advantages and disadvantages of remote working."
  },
  "scores": {
    "cefr_band": "B2",
    "cefr_continuous": 4.28,
    "confidence_interval_95": [4.12, 4.44]
  },
  "quadrant_breakdown": {
    "pronunciation": {
      "overall_gop_accuracy": 82.4,
      "indian_allophone_tolerance_applied": true,
      "allophones_detected": ["T_RETROFLEX", "T_DENTAL", "V_APPROX"]
    },
    "fluency": {
      "speech_rate_sps": 4.12,
      "articulation_rate_sps": 4.78,
      "pause_to_speech_ratio": 0.138,
      "mean_run_length_syllables": 7.4
    },
    "grammar": {
      "max_dependency_tree_depth": 5,
      "mean_tree_depth": 3.8,
      "subordinate_clause_density": 1.67
    },
    "vocabulary_and_coherence": {
      "task_relevance_cosine": 0.884,
      "inter_sentence_coherence": 0.712,
      "lexical_distribution": {
        "A1_A2": 0.58,
        "B1_B2": 0.31,
        "C1_C2": 0.11
      }
    }
  },
  "transcript": {
    "raw": "although working from home saves commute time we must coordinate properly",
    "punctuated": "Although working from home saves commute time, we must coordinate properly."
  },
  "phoneme_diagnostics": [
    {
      "token": "th",
      "canonical_target": "DH",
      "matched_allophone": "D_DENTAL",
      "gop_score": 91.2,
      "status": "GREEN"
    },
    {
      "token": "t",
      "canonical_target": "T",
      "matched_allophone": "T_RETROFLEX",
      "gop_score": 88.6,
      "status": "GREEN"
    },
    {
      "token": "r",
      "canonical_target": "R",
      "matched_allophone": "R_FLAP",
      "gop_score": 41.3,
      "status": "RED"
    }
  ]
}