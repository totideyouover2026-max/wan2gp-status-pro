Yes. I traced the current `main` branch rather than relying on the model descriptions. Wan2GP is currently **v13.14**, and the important finding is that the 200+ default presets collapse into a much smaller number of distinct generation pipelines. Quantisation, FastWan/Lightning/distillation variants, LoRAs, attention modes and cache settings generally **do not change which stages occur**.

I’d use these stage names for Wan2GP:

**✓ = normally/always occurs · ◐ = occurs only for certain inputs/settings · — = not used**

### Video / image models

| Model / stage-equivalent presets | Input VAE / media encode | Prompt Encoding | Denoising | AR / Tokens | Native extra pass | VAE / media Decode | Output Enhancement |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Wan T2V** — `t2v`, `t2v_1.3B`, `t2v_2_2`, SF/FusionX/Nexus/NVFP4 etc. | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Wan I2V / TI2V / FLF2V** — `i2v`, `i2v_720p`, `i2v_2_2`, SVI2Pro, Enhanced Lightning, FusionX, `ti2v_2_2`, `flf2v_720p` | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Wan VACE / control / edit** — VACE variants, Fun InP, Phantom, ReCam, Animate, Alpha, Chrono Edit, Lucy Edit, Kiwi Edit, StandIn, Lynx, Fantasy, Mocha, SteadyDancer, WanMove, SCAIL, Vista4D, Bernini | ◐ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Wan MultiTalk / InfiniteTalk** | ✓¹ | ✓ | ✓ | — | — | ✓ video | ◐ |
| **SkyReels Diffusion Forcing** | ◐ | ✓ | ✓ | — | — | ✓ | ◐ |
| **OVI / OVI 1.1 / FastWan** | ◐ | ✓ | ✓ **video + audio** | — | — | ✓ **video + audio** | ◐ |
| **Hunyuan T2V** | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Hunyuan I2V / Custom / Edit / Avatar** | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Hunyuan 1.5 T2V** | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Hunyuan 1.5 I2V** | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Hunyuan 1.5 Upsampler / 1080** | ✓ | ✓ | ✓ | — | ✓ **SR** | ✓ | — |
| **LTX Video 0.9.x** — `ltxv_13B`, distilled | ◐ | ✓ | ✓ | — | ✓ **multiscale / latent upscale** | ✓ | ◐ |
| **LTX2 / 2.3 / 2.5** — 19B, 22B, 25 22B, distilled, GGUF, NVFP4 | ◐ | ✓ | ✓ **A/V** | — | ◐ **stage 2** | ✓ **video + audio** | ◐ |
| **LTX2 Edit Anything / MSR / JoyAI Echo** | ✓ | ✓ | ✓ **A/V** | — | ◐ **stage 2** | ✓ **video + audio** | ◐ |
| **LongCat Video** | ◐ | ✓ | ✓ | — | — | ✓ | ◐ |
| **LongCat Avatar / Avatar 1.5** | ✓¹ | ✓ | ✓ | — | — | ✓ | ◐ |
| **MiniMax H3 FL2VA / Pruned / PDD / VDN** | ◐ | ✓ | ✓ **video + audio** | — | ◐ | ✓ **video + audio** | ◐ |
| **MiniMax H3 Ref2VA / Pruned / PDD** | ◐² | ✓ | ✓ **video + audio** | — | ◐ | ✓ **video + audio** | ◐ |
| **H3 Viggle Animate** | ✓ | —³ | ✓ **video + audio** | — | — | ✓ **video + audio** | ◐ |
| **MAGI Human / Distill** | ◐¹ | ✓ | ✓ **video + audio** | — | — | ✓ **video + audio** | ◐ |
| **MAGI Human SR1080 variants** | ◐¹ | ✓ | ✓ **video + audio** | — | ✓ **second SR diffusion** | ✓ **video + audio** | ◐ |
| **Kandinsky 5 T2V** — Lite/Pro/distilled/sparse | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Kandinsky 5 I2V** | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Flux / Flux Schnell / Krea / Chroma** | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Flux Kontext / UMO / USO** | ✓/◐ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Flux 2 / Klein / Klein Base / Pi Flux 2** | ◐ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Qwen Image** — 20B/2512/2.1 | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Qwen Image Edit / Edit Plus / Plus2** | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Qwen Image Layered** | ✓ | ✓ | ✓ | — | — | ✓ + layer assembly | ◐ |
| **Krea 2 Raw / Turbo** | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Krea 2 Raw Edit / Turbo Edit** | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Z-Image / Base / TwinFlow** | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Z-Image Control / Control2 / 2.1** | ✓ | ✓ | ✓ | — | — | ✓ | ◐ |
| **HiDream O1 / O1 Dev** | ◐ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Ideogram 4 / TurboTime / NF4** | — | ✓ | ✓ | — | — | ✓ | ◐ |
| **Ming Image Design** | ◐⁴ | ✓ | ✓ | — | — | ✓ | ◐ |
| **Ming Image Design Layer** | ✓⁴ | ✓ | ✓ | — | — | ✓ + layer assembly | ◐ |
| **SenseNova U1** | ◐ | ✓ | ✓ | — | — | ✓ | ◐ |

¹ Not necessarily a VAE: audio/avatar conditioning has its own feature encoder.  
² Ref2VA may encode images through vision/conditioning paths as well as video/audio VAEs, so “Input VAE” is too narrow a name.  
³ Viggle explicitly disables the normal text encoder and ignores the generation prompt; it uses its fixed internal instruction.  
⁴ Ming uses a multimodal/vision encoder for reference images rather than treating every input as a conventional VAE encode.

### H3 deserves a more detailed split

H3 is particularly relevant because its pipeline can genuinely change shape according to settings:

| H3 setting | Input Encode | Prompt Encode | Denoise 1 | Latent Upscale | Denoise 2 | Audio Refinement | Video Decode | Audio Decode |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard one-phase | ◐ | ✓ | ✓ | — | — | — | ✓ | ✓ |
| **Two-phase high resolution** | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| **Audio Refinement enabled** | ◐ | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| Two-phase + Audio Refinement | ◐ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| H3 Text-to-Image mode | ◐ | ✓ | ✓ | ◐ | ◐ | — | ✓ | — |
| H3 Audio-only Ref2VA | ◐ | ✓ | ✓⁵ | — | — | ◐ | **—** | ✓ |

⁵ The audio-only H3 model still jointly denoises a tiny internal 32×32 video latent, but Wan2GP deliberately **does not VAE-decode that video**.

So for a full H3 two-phase + audio refinement run, the conceptual sequence is:

**Input/media encoding → Prompt encoding → Phase 1 denoising → latent/high-resolution preparation → Phase 2 denoising → Audio Refinement denoising → Video VAE decoding → Audio VAE decoding → optional post-processing → mux/save.**

That matches what the H3 code actually does, including statuses such as `Phase 2/2 High Resolution`, `Audio Refinement Extra Phase`, `VAE Decoding of Video and Audio`, and `Decoding H3 Stereo Audio`.

### Audio / TTS / music models

These are why I wouldn't hard-code every Wan2GP model as `Prompt → Denoise → VAE Decode`. Several don't diffuse at all.

| Model | Input / Voice Encode | Prompt / Text Encode | AR / Token Generation | Denoising / Flow | Codec / VAE / Vocoder Decode | Extra |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **H3 Audio** | ◐ | ✓ | — | ✓ | ✓ | ◐ Audio Refinement |
| **Scenema Audio** | ◐ | ✓ | — | ✓ | ✓ | ◐ SeedVC voice processing |
| **DramaBox Audio** | ◐ | ✓ | — | ✓ | ✓ | — |
| **ACE-Step v1** | ◐ | ✓ | — | ✓ | ✓ | — |
| **ACE-Step 1.5 / XL** | ◐ | ✓ | ◐⁶ | ✓ | ✓ | — |
| **Chatterbox** | ◐ | ✓ | ✓ | ✓ acoustic | ✓ | — |
| **Qwen3 TTS Base / CustomVoice / VoiceDesign** | ◐ | ✓ | ✓ | **—** | ✓ speech codec | — |
| **IndexTTS2 / 2.5** | ✓ | ✓ | ✓ | ✓ acoustic | ✓ BigVGAN | — |
| **HeartMula** | ◐ | ✓ | ✓ | ✓ codec/acoustic | ✓ | — |
| **KugelAudio** | ◐ | ✓ | ✓ | ✓ diffusion head | ✓ | — |
| **Stable Audio 3 Small / Medium / SFX** | ◐ | ✓ | — | ✓ | ✓ | — |
| **OmniVoice** | ✓/◐ | ✓ | ✓ | — | ✓ audio tokenizer | — |
| **MiniMax Music 3** | ◐ | ✓ | ✓ | ✓ acoustic/flow | ✓ vocoder | — |
| **AuK / AuK Flash** | ◐ | ✓ | — | ✓ | ✓ VAE | — |
| **YuE2** | ◐ | ✓ | ✓ **composition** | ✓ **acoustic** | ✓ VAE | — |
| **YuE2 Hum** | ✓ | ✓ | ✓ **composition** | ✓ **acoustic** | ✓ VAE | Hum encoding |

⁶ ACE-Step 1.5 LM variants add the language-model conditioning/generation stage; it isn't equivalent to another diffusion pass.

---

### The stage vocabulary I'd actually use

For something like your Status Pro work, I'd expand your current set slightly. Wan2GP itself already explicitly instruments **text encoding, VAE encoding and VAE decoding** in [`shared/utils/phase_progress.py`](https://github.com/deepbeepmeep/Wan2GP/blob/main/shared/utils/phase_progress.py), and the generic UI distinguishes multiple denoising phases.

I think the clean canonical set is:

| Canonical stage | Covers |
|---|---|
| **Input Processing** | resize/crop, masks, control extraction, audio preparation |
| **Input Encoding** | Video/Image VAE encode, vision encode, audio VAE/codec/features |
| **Prompt Encoding** | T5/Qwen/Gemma/etc. embedding generation |
| **Token Generation** | AR/LM stage in Qwen TTS, YuE2, Music3, Chatterbox etc. |
| **Denoising** | Main diffusion/flow sampling |
| **Latent Upsampling** | LTX/H3 native resolution transition |
| **Refinement** | H3 Audio Refinement, second SR diffusion, model-native refinement |
| **VAE / Codec Decoding** | Video/image VAE, audio VAE, vocoder or codec |
| **Enhancement** | FlashVSR/DLSS/etc. and other selected post-generation spatial enhancement |
| **Temporal Enhancement** | RIFE/frame interpolation etc. |
| **Audio Post-processing** | voice replacement, soundtrack processing etc. |
| **Assembly / Save** | sliding-window joining, audio/video muxing, encode/write |

Two particularly important conclusions for your status-stage logic:

**“Input VAE” should probably become “Input Encoding”.** H3 Ref2VA, MAGI, MultiTalk, Ming, TTS and several others perform important input encodes that aren't technically VAEs.

And **“Enhancement” should not be considered a normal mandatory model stage**. There are three different things currently hiding under that idea: optional **Prompt Enhancement before generation**, model-native **latent/SR refinement inside generation**, and optional **Post Processing after decoding**. They should be tracked separately.

The core code I used for the audit was `wgp.py`, `shared/utils/phase_progress.py`, `models/wan/any2video.py`, `models/wan/ovi_fusion_engine.py`, `models/minimax_h3/pipeline.py`, the LTX2 one/two-stage pipelines, `models/magi_human/magi_human_model.py`, plus the individual image/audio handlers and pipelines.

If this is ultimately for **Status Pro/Lite**, the next useful step would be to turn this into a machine-friendly mapping such as `architecture → possible stages → conditions`, rather than maintaining 200 individual model names. That would also automatically handle new finetunes inheriting an existing architecture.