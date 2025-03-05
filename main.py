import random
import os
import shutil
import time

from pathlib import Path
from typing import Generator

import gradio as gr
import librosa
import numpy as np
import scipy.io as sio
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    pipeline,
    TextIteratorStreamer,
)

import speech_enhancement
import speaker_recognition
import text_generation
import text_to_speech

from final_v1 import speaker_detect
from feature_extraction import extract_MFB_enroll

PATH = Path(__file__).parent.resolve()

#=============================================================================================
# 數位訊號處理
DNS_64_URL = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th"
se_model = speech_enhancement.from_pretrained(DNS_64_URL)
se_model.eval()
#=============================================================================================
# 語者辨識
TASK = "speaker-recognition"
model_path = Path.joinpath(PATH, TASK.replace("-", "_"), "resnet", "checkpoint_702.pth")
sr_model = speaker_recognition.from_pretrained(model_path, device="cuda")
sr_model.eval()
#=============================================================================================
# 語音辨識
pipelines: dict[str, Pipeline] = {}

TASK = "automatic-speech-recognition"
model_path = Path.joinpath(PATH, TASK.replace("-", "_"), "whisper-large-v3")
pipelines[TASK] = pipeline(
    TASK,
    model=model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    model_kwargs={
        # only be used for models with the fp16 or bf16 torch type
        "attn_implementation": "flash_attention_2",
    },
)
#=============================================================================================
# 自然語言處理
TASK = "text-generation"
model_path = Path.joinpath(
    PATH, TASK.replace("-", "_"), "Llama3-8B", "Llama3-8B-Chinese-Chat"
)
# 做量化處理
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
# 將模型和量化處理同時載入
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.eos_token_id = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
pipelines[TASK] = pipeline(
    TASK,
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
    streamer=streamer,
)
#=============================================================================================
# 語音合成
checkpoint_dir = Path.joinpath(PATH, "text_to_speech", "XTTS-v2")
xtts_config, xtts_model = text_to_speech.from_pretrained(checkpoint_dir, device="cuda")

#=============================================================================================
# 主程式(應用程式介面、各個模組的呼叫和使用)
def main():
    theme = gr.themes.Default().set(
        loader_color="#FF0000",
        slider_color="#000000",
    )

    with gr.Blocks(theme) as demo:
        gr.Markdown(
            """
            # BrasLab 數位語音助理系統之實作與實現 Implementation and Realization of Digital Voice Assistant System
        
            # 指導教授：許正欣  , 研究生：林建宏 Jerry_Lin
            """
        )

        with gr.Row():
            with gr.Tabs():
                # ======================================使用版面======================================
                with gr.TabItem("Inference"):
                    gr.Markdown(
                        """
                        1. 請錄製一段2秒的音檔，以使用語音助理系統。
                        """
                    )
                    input_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        label="Input audio",
                        min_length=2,
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            show_controls=False,
                            skip_length=2,
                            sample_rate=16000,
                        ),
                    )
                    gr.Markdown(
                        """
                        2. 可透過拉動下方的滑桿，以調整降噪的閾值。
                        """
                    )
                    snr_slider = gr.Slider(
                        -100,
                        100,
                        step=1,
                        value=15,
                        label="SNR threshold",
                        info="Choose between -100 and 100 dB",
                    )
                    gr.Markdown(
                        """
                        3. （可選）語音合成複製音訊（最短6秒）。
                        """
                    )
                    clone_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Clone audio",
                        min_length=6,
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            show_controls=False,
                            skip_length=2,
                            sample_rate=16000,
                        ),
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.ClearButton([input_audio, clone_audio])

                        with gr.Column():
                            input_btn = gr.Button("Submit", variant="primary")
                # ======================================註冊本系統======================================
                with gr.TabItem("Register"):
                    gr.Markdown(
                        """
                        # ✯ 使用者註冊 (User Sign-up)

                        如果為第一次使用本系統的使用者，請於下方進行註冊。
                        
                        1. 請先輸入名稱至下方文字方塊中。
                        """
                    )
                    # ================輸入名稱================
                    register_name = gr.Textbox(label="Name")
                    gr.Markdown(
                        """
                        2. 請錄製一段30秒的音檔，以便註冊您的聲紋。
                        """
                    )
                    # ================錄音程式、錄音 or 上傳音檔================
                    register_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        label="Register audio",
                        min_length=10,
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#8600FF",
                            waveform_progress_color="#8600FF",
                            show_controls=False,
                            skip_length=2,
                            sample_rate=16000,
                        ),
                    )
                    gr.Markdown(
                        """
                        3. 請記得提交，當看到 Successfull user name 時代表註冊成功。
                        """
                    )
                    register_result = gr.Textbox(value="", label="Register result")

                    with gr.Row():
                        with gr.Column():
                            gr.ClearButton(register_audio)

                        with gr.Column():
                            register_btn = gr.Button("Submit", variant="primary")
                            
                    register_btn.click(
                        # ================做音檔的預先處理(多轉單、下採樣)================
                        fn=preprocess, inputs=register_audio, outputs=register_audio
                    ).then(
                        # ================語者註冊================
                        fn=speaker_register,
                        inputs=[register_audio, register_name],
                        outputs=register_result,
                    ).then(
                        fn=lambda: None, inputs=[], outputs=register_audio
                    )
            # =============================================================================================
            with gr.Column():
                gr.Markdown(
                    """
                    輸出結果如下所示：
                    """
                )
                with gr.Row():
                    with gr.Column():
                        # ================版面設計、對話框不提供輸入================
                        enhanced_audio = gr.Audio(
                            label="Enhanced audio", interactive=False
                        )
                        snr = gr.Textbox(label="SNR", interactive=False)
                        user = gr.Textbox(label="User", interactive=False)
                        asr_result = gr.Textbox(label="Input", interactive=False)
                        llm_result = gr.Textbox(label="Output", interactive=False)
                        synthesized_audio = gr.Audio(
                            label="Synthesized audio",
                            interactive=False,
                            streaming=True,
                            autoplay=True,
                        )
                        # ================計算程式整體執行的時間================
                        start_time = gr.Number(visible=False)
                        elapsed_time = gr.Textbox(
                            label="Elapsed time", interactive=False
                        )
                        input_btn.click(
                            fn=lambda: time.time(), inputs=[], outputs=start_time
                        ).then(
                            # 音檔預處理 -> 單通道、下採樣
                            fn=preprocess,
                            inputs=input_audio,
                            outputs=input_audio,
                        ).then(
                            # 裡面有計算snr的公式 -> 降造 or 不降造 音檔 & 該音檔 snr 值
                            fn=enhance,
                            inputs=[input_audio, snr_slider],
                            outputs=[
                                enhanced_audio,
                                snr,
                            ],
                        ).then(
                            fn=lambda: None,
                            inputs=[],
                            outputs=[input_audio],
                        ).then(
                            # 語者辨識 -> 語者名稱
                            fn=recognize, inputs=[enhanced_audio], outputs=[user]
                        ).then(
                            # 語音、使用者名稱 -> 文字
                            fn=transcribe,
                            inputs=[enhanced_audio, user],
                            outputs=asr_result,
                        ).then(
                            # 文字 -> 回復訊息生成
                            fn=generate,
                            inputs=[asr_result, user],
                            outputs=llm_result,
                        ).then(
                            # 回復訊息文字 -> 音檔
                            fn=synthesize,
                            inputs=[llm_result, user, clone_audio],
                            outputs=synthesized_audio,
                        ).then(
                            fn=lambda: None,
                            inputs=[],
                            outputs=[clone_audio],
                        ).then(
                            # 計算程式總執行時間
                            fn=lambda start_time: f"{time.time() - start_time} s",
                            inputs=start_time,
                            outputs=elapsed_time,
                        )
    # 啟動伺服器(不限網域、http to https)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=6006,
        ssl_verify=False,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
    )

#===================================================================================
# 音檔預先處理
def preprocess(audio: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    sample_rate, data = audio
    data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim == 2:
        data = data.swapaxes(0, 1)
    # 多通道 轉 單通道
    mono_audio = librosa.to_mono(data)
    # 強制轉型太
    mono_audio = librosa.resample(mono_audio, orig_sr=sample_rate, target_sr=16000)
    return (16000, mono_audio.ravel())
#===================================================================================
# 訊號處理含式
def enhance( audio: tuple[int, np.ndarray], snr_threshold: int) -> tuple[tuple[int, np.ndarray], str]:
    snr, enhanced_audio = speech_enhancement.try_enhancement(
        se_model, audio=audio[1], snr_threshold=snr_threshold
    )
    return ((16000, enhanced_audio), f"{snr} dB")
#===================================================================================
# 自然語言處理
def generate(inputs: str, speaker: str) -> Generator:
    if speaker == "Imposter":
        yield "很抱歉，您不是註冊語者，我無法為您提供服務。"
    else:
        text_generation.generate(
            pipelines["text-generation"],
            inputs,
            do_sample=True,
            num_return_sequences=1,
            max_length=1024,
            repetition_penalty=1.1,
            temperature=0.2,
            truncation=True,
            top_k=50,
            top_p=0.9,
        )

        token_strs = []
        for token_str in streamer:
            token_strs.append(token_str)
            yield "".join(token_strs)
#===================================================================================
# 語者辨識
def recognize(audio: tuple[int, np.ndarray]) -> str:
    return speaker_detect(sr_model, audio[1])
#===================================================================================
# 語音合成
def synthesize(inputs: str, speaker: str, speaker_wav: str = None) -> Generator:
    if speaker == "Imposter":
        inputs = "很抱歉，您不是註冊語者，我無法為您提供服務。"

    if not speaker_wav:
        speaker_wav = list(Path.joinpath(PATH, "text_to_speech", "speakers").iterdir())
    chunks = text_to_speech.synthesize(xtts_config, xtts_model, inputs, speaker_wav)

    for chunk in chunks:
        yield (24000, chunk.cpu().numpy())

#===================================================================================
# 語音辨識
def transcribe(audio: tuple[int, np.ndarray], speaker: str) -> str:
    if speaker == "Imposter":
        return "Reject"

    float_audio = audio[1].astype(np.float32) / 32767.0
    return pipelines["automatic-speech-recognition"](
        float_audio, generate_kwargs={"language": "mandarin"}
    )["text"]
#===================================================================================
# 註冊語者再用的程式
def speaker_register(input_audio, register_name):
    # 下採樣
    _, data = input_audio
    audio = data

    total = len(audio)

    end = 3 * 16000

    available_duration = total - end

    # 計算資料夾數量
    if os.path.exists("./data/enroll/.ipynb_checkpoints"):
        shutil.rmtree("./data/enroll/.ipynb_checkpoints")

    num_enroll_utterance = 6
    for i in range(num_enroll_utterance):
        # 起始
        start_sample = random.randint(0, available_duration)
        # 結尾
        end_sample = start_sample + end
        # [起始、結尾] = 區間
        segment = audio[start_sample:end_sample]

        file_name_1 = "./data/record/" + str(f"segment1_{i}.wav")

        sio.wavfile.write(file_name_1, 16000, segment)

        extract_MFB_enroll(segment, sr_model, i, register_name)

    return "Successfull " + str(register_name)

#===================================================================================
if __name__ == "__main__":
    main()
