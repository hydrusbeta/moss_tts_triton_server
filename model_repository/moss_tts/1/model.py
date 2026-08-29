# Quick copyright/license note
# Much of the code in this file is taken from the following repo and modified:
# https://huggingface.co/ZDisket/MOSS-TTS-PNY/blob/main/moss_tts_torchopt_runner_bundle/portable_tts_runtime.py
# That repo is owned by ZDisket/Delta, who communicated to me that their code is open source and may be redistributed.
# I suspect that ZDisket referenced some sample code from the following repo while writing their own code:
# https://huggingface.co/OpenMOSS-Team/MOSS-TTS
# As of the time of this writing, that code is licensed under Apache 2.0.

import os.path
import gc
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoConfig, AutoTokenizer, AutoModel

from moss_tts_local_clipper_checkpoint.processing_moss_tts import MossTTSDelayProcessor
from moss_tts_torchopt_runner_bundle.portable_tts_runtime import OptimizedTTSConfig
from moss_tts_torchopt_runner_bundle.decoder4_features_torch import Decoder4FeatureExtractor

# helper for running the vocoder
def run_vocoder_onnx(session: Any, features: np.ndarray, feature_lengths: np.ndarray) -> torch.Tensor:
    input_name = session.get_inputs()[0].name
    audio_np = session.run(None, {input_name: features.astype(np.float32, copy=False)})[0]
    samples = int(feature_lengths[0]) * 960
    audio = torch.from_numpy(audio_np[0, 0, :samples]).float().cpu()
    return torch.nan_to_num(audio).reshape(-1).clamp(-1.0, 1.0)

# helper for running the vocoder
def run_vocoder(
    session: Any,
    features: np.ndarray | torch.Tensor,
    feature_lengths: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    if isinstance(session, TorchScriptVocoderRuntime):
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).to(dtype=torch.float32)
        else:
            features_tensor = features.to(dtype=torch.float32)
        with torch.inference_mode():
            audio_tensor = session.run_tensor(features_tensor)
        if isinstance(feature_lengths, torch.Tensor):
            samples = int(feature_lengths[0].item()) * 960
        else:
            samples = int(feature_lengths[0]) * 960
        audio = audio_tensor[0, 0, :samples].detach().float().cpu()
        return torch.nan_to_num(audio).reshape(-1).clamp(-1.0, 1.0)

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(feature_lengths, torch.Tensor):
        feature_lengths = feature_lengths.detach().cpu().numpy()
    return run_vocoder_onnx(session, features, feature_lengths)

def enable_static_cache_for_global_model(model: Any) -> None:
    type(model)._can_compile_fullgraph = True
    language_config = model.config.language_config
    for field in (
        "max_position_embeddings",
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "num_key_value_heads",
        "sliding_window",
        "layer_types",
        "num_hidden_layers",
    ):
        if hasattr(language_config, field):
            setattr(model.config, field, getattr(language_config, field))

def tensorize_rmsnorm_eps(model):
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        for attr in ("variance_epsilon", "eps"):
            value = getattr(module, attr, None)
            if isinstance(value, (float, int)):
                setattr(module, "_torchopt_rmsnorm_eps_float", float(value))
                setattr(
                    module,
                    attr,
                    torch.tensor(float(value), dtype=torch.float32),
                )

class TritonPythonModel:
    def initialize(self, args):
        # 0. Config and paths
        model_root = (os.path.dirname(os.path.abspath(__file__)))
        moss_tts_checkpoint = os.path.join(model_root, "moss_tts_local_clipper_checkpoint")
        tokenizer_path = os.path.join(model_root, 'moss_audio_tokenizer')
        vocoder_path = os.path.join(model_root, 'istftnet2_decoder4_50hz')
        vocoder_checkpoint = os.path.join(vocoder_path, 'istftnet2_decoder_cpu.ts')
        self.moss_tts_config = OptimizedTTSConfig(
            checkpoint=moss_tts_checkpoint,
            codec_path=tokenizer_path,
            decoder_dir=vocoder_path,
            decoder4_features_onnx=vocoder_checkpoint,
            decoder_runtime='torchscript_cpu',
            vocoder_cudagraph=False,
            vocoder_bucket_frames=64,
            vocoder_prewarm_buckets='',
            decoder4_features_runtime='torch_fp16',
            decoder4_provider='CPUExecutionProvider',
            dtype='fp16',
            tts_quantization='none',
            torch_opt_mode='none',
            compile_mode='max-autotune-no-cudagraphs',
            cache_implementation='static',
            compile_global_transformer=False,
            global_compile_mode='default',
            attn_implementation='sdpa',
            style_bert_model='cirimus/modernbert-base-go-emotions',
            style_bert_layer=19,
            style_bert_max_length=512,
            fast_prepare_inputs=True,
            triton_top_p=True,
            triton_fused_lm_head=False,
            triton_qkv_cache=False,
            packed_local_qkv=True,
            packed_local_mlp=True,
            packed_adapter_mlp=False,
            packed_adapter_mlp_scope='heads',
            static_packed_weights=True,
            triton_rmsnorm=False,
            tensorize_rmsnorm_eps=True,
            fast_control_head=False,
            feedback_lookup=True,
            local_compile_fullgraph=False,
        )

        # 1. Initialize the text tokenizer
        tokenizer_config = AutoConfig.from_pretrained(moss_tts_checkpoint, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(moss_tts_checkpoint, trust_remote_code=True)
        self.processor = MossTTSDelayProcessor(tokenizer=tokenizer, model_config=tokenizer_config)
        self.processor.model_config.sampling_rate = 48000

        # 2. Initialize the token -> RVQ_code model
        tts_load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "attn_implementation": self.moss_tts_config.attn_implementation,
        }
        self.model = AutoModel.from_pretrained(moss_tts_checkpoint, **tts_load_kwargs)
        tensorize_rmsnorm_eps(self.model)
        enable_static_cache_for_global_model(self.model)

        # 3. Initialize the audio tokenizer
        self.decoder4_session = TorchDecoder4FeatureRuntime(self.moss_tts_config.codec_path, dtype=torch.float16)

        # 4. Initialize the vocoder
        self.vocoder_session = TorchScriptVocoderRuntime(self.moss_tts_config.decoder_dir)


    def execute(self, requests):
        pass
        responses = []
        for request in requests:
            # Parse inputs:
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0].decode("utf-8")
            print(text, flush=True)
            n_vq = pb_utils.get_input_tensor_by_name(request, "n_vq").as_numpy()[0]
            audio_temperature = pb_utils.get_input_tensor_by_name(request, "audio_temperature").as_numpy()[0]
            audio_top_p = pb_utils.get_input_tensor_by_name(request, "audio_top_p").as_numpy()[0]
            audio_top_k = pb_utils.get_input_tensor_by_name(request, "audio_top_k").as_numpy()[0]
            repetition_penalty = pb_utils.get_input_tensor_by_name(request, "repetition_penalty").as_numpy()[0]
            speaker_id = pb_utils.get_input_tensor_by_name(request, "speaker_id").as_numpy()[0]
            language = pb_utils.get_input_tensor_by_name(request, "language").as_numpy()[0].decode("utf-8")
            emotion_id = pb_utils.get_input_tensor_by_name(request, "emotion_id").as_numpy()[0] # Originally an int that is cast to a float and concatenated with emotion_energy
            emotion_energy = pb_utils.get_input_tensor_by_name(request, "emotion_energy").as_numpy()[0]
            max_new_tokens = pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy()[0]

            # 1. Tokenize the text input
            conversations = [[self.processor.build_user_message(text=text, language=language)]]
            batch = self.processor(conversations, mode="generation")
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # 2. Convert the text tokens to audio tokens
            generation_kwargs = {
                "cache_implementation" : self.moss_tts_config.cache_implementation,
                "speaker_ids" : torch.tensor([speaker_id], dtype=torch.long),
                "style_features": torch.tensor(
                    [
                        [
                            float(emotion_id), # an integer from 0 to self.model.config['num_emotions']
                            float(emotion_energy), # a float between 0 and 1
                        ]
                    ],
                    dtype=torch.float32,
                )
            }
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    n_vq_for_inference=n_vq,
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=repetition_penalty,
                    **generation_kwargs,
                )

            # 3, 4. Decode the audio tokens and use the vocoder to convert them to a wave form
            audio = self.decode_outputs(outputs)
            audio = (audio.clamp(-1.0, 1.0) * 32767.0).numpy().astype(np.int16)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "audio", audio
                    )
                ]
            )
            responses.append(inference_response)

            # Note: The caller should write the output to a wav file with 16 bits per sample and 48,000 samples per second:
            #     with wave.open(str(path), "wb") as handle:
            #         handle.setnchannels(1)
            #         handle.setsampwidth(2)
            #         handle.setframerate(48000)
            #         handle.writeframes(audio.tobytes())
        return responses


    def decode_outputs(self, outputs: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decoded_segments: list[torch.Tensor] = []
        codes_input_name = self.decoder4_session.get_inputs()[0].name
        lengths_input_name = self.decoder4_session.get_inputs()[1].name
        for start_length, generation_ids in outputs:
            frame_tokens = generation_ids.detach().cpu()[:, 0]
            audio_codes = generation_ids.detach().cpu()[:, 1:]
            is_pad = (audio_codes == int(self.processor.model_config.audio_pad_code)).all(dim=1)
            is_eos = frame_tokens == int(self.processor.model_config.audio_end_token_id)
            non_pad = ~is_pad & ~is_eos
            if not non_pad.any():
                continue
            idx = torch.nonzero(non_pad).squeeze(1)
            breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
            segments_idx = [idx] if breaks.numel() == 0 else list(torch.split(idx, breaks.tolist()))
            for segment_index, segment_idx in enumerate(segments_idx):
                segment_codes = audio_codes[segment_idx].contiguous()
                if int(segment_codes.shape[0]) <= 0:
                    continue
                codes_np = segment_codes.T.unsqueeze(1).numpy().astype(np.int64, copy=False)
                lengths_np = np.asarray([int(segment_codes.shape[0])], dtype=np.int64)
                feeds = {codes_input_name: codes_np, lengths_input_name: lengths_np}
                if hasattr(self.decoder4_session, "run_tensors"):
                    features, feature_lengths = self.decoder4_session.run_tensors(feeds)
                else:
                    features, feature_lengths = self.decoder4_session.run(None, feeds)
                segment_audio = run_vocoder(self.vocoder_session, features, feature_lengths)
                if segment_index == 0 and int(start_length) > 0:
                    trim_ratio = max(0.0, min(float(start_length) / float(segment_codes.shape[0]), 1.0))
                    if trim_ratio >= 1.0:
                        continue
                    if trim_ratio > 0.0:
                        segment_audio = segment_audio[..., int(segment_audio.shape[-1] * trim_ratio) :]
                decoded_segments.append(segment_audio)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not decoded_segments:
            raise RuntimeError("Generation did not produce decodable audio.")
        return torch.cat(decoded_segments, dim=-1).float().cpu()

class TorchDecoder4FeatureRuntime:
    def __init__(self, codec_path: str, dtype: torch.dtype) -> None:
        audio_tokenizer = AutoModel.from_pretrained(
            codec_path,
            trust_remote_code=True,
            dtype=dtype)
        audio_tokenizer.eval()
        num_quantizers = int(audio_tokenizer.config.quantizer_kwargs.get("num_quantizers", 32))
        self.extractor = Decoder4FeatureExtractor(
            audio_tokenizer,
            num_quantizers=num_quantizers,
            output_dtype=dtype,
        )
        self.extractor.eval()
        self.inputs = [SimpleNamespace(name="codes"), SimpleNamespace(name="lengths")]
        del audio_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self) -> list[Any]:
        return self.inputs

    def run(self, output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names
        codes = torch.from_numpy(feeds["codes"])
        lengths = torch.from_numpy(feeds["lengths"])
        with torch.inference_mode():
            features, feature_lengths = self.extractor(codes, lengths)
        return [
            features.detach().float().cpu().numpy(),
            feature_lengths.detach().cpu().numpy(),
        ]

class TorchScriptVocoderRuntime:
    def __init__(
        self,
        decoder_dir: str,
        *,
        use_cudagraph: bool = False,
        bucket_frames: int = 0,
    ) -> None:
        artifact = os.path.join(decoder_dir, 'istftnet2_decoder_cpu.ts')
        if not os.path.exists(artifact):
            raise FileNotFoundError(f"Missing TorchScript vocoder artifact: {artifact}")
        self.module = torch.jit.load(artifact, map_location="cpu").eval()
        self.use_cudagraph = False
        self.bucket_frames = max(0, int(bucket_frames))
        self.graphs: dict[tuple[Any, ...], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="features")]

    def prewarm_buckets(self, frame_lengths: list[int]) -> dict[str, Any]:
        requested = [int(length) for length in frame_lengths if int(length) > 0]
        if not requested:
            return {"requested": [], "elapsed_sec": 0.0}
        start = time.perf_counter()
        warmed: list[int] = []
        with torch.inference_mode():
            for frames in requested:
                features = torch.randn(1, 768, frames, dtype=torch.float32)
                self.run_tensor(features)
                warmed.append(frames)
        return {"requested": requested, "warmed": warmed, "elapsed_sec": time.perf_counter() - start}

    def run_tensor(self, features_tensor: torch.Tensor) -> torch.Tensor:
        features_tensor = features_tensor
        if self.bucket_frames > 0 and features_tensor.ndim == 3:
            frames = int(features_tensor.shape[-1])
            bucketed = ((frames + self.bucket_frames - 1) // self.bucket_frames) * self.bucket_frames
            if bucketed > frames:
                padded = torch.zeros(
                    *features_tensor.shape[:-1],
                    bucketed,
                    dtype=features_tensor.dtype,
                )
                padded[..., :frames] = features_tensor
                features_tensor = padded
        if not self.use_cudagraph:
            with torch.inference_mode():
                return self.module(features_tensor)
        key = (
            features_tensor.device.index,
            features_tensor.dtype,
            tuple(features_tensor.shape),
        )
        entry = self.graphs.get(key)
        if entry is None:
            try:
                static_features = torch.empty_like(features_tensor)
                static_features.copy_(features_tensor)
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream), torch.inference_mode():
                    for _ in range(3):
                        static_audio = self.module(static_features)
                torch.cuda.current_stream().wait_stream(warmup_stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    static_audio = self.module(static_features)
                entry = (graph, static_features, static_audio)
                self.graphs[key] = entry
            except Exception:
                self.use_cudagraph = False
                self.graphs.clear()
                with torch.inference_mode():
                    return self.module(features_tensor)

        graph, static_features, static_audio = entry
        static_features.copy_(features_tensor)
        graph.replay()
        return static_audio
