# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import platform  # 用于检测操作系统
import subprocess  # 用于打开文件目录
from funasr import AutoModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
)



from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

# ===================== 辅助功能函数 =====================
def generate_seed():
    """生成随机种子"""
    return random.randint(1, 2**32 - 1)


def change_instruction(mode_checkbox_group):
    """根据选择的模式更新操作说明"""
    return instruct_dict[mode_checkbox_group]


# 修复重复音色列表问题
def refresh_sft_spk():
    """刷新音色选择列表 - 修复重复音色问题并自动注册新文件"""
    try:
        # 确保音色信息是最新的
        if hasattr(cosyvoice.frontend, 'load_spkinfo'):
            cosyvoice.frontend.load_spkinfo()

        # 获取当前已注册的音色列表
        current_choices = cosyvoice.list_available_spks()

        # 扫描自定义音色目录，检测新文件
        custom_voices_dir = os.path.join(cosyvoice.model_dir, 'custom_voices')
        if os.path.exists(custom_voices_dir):
            # 获取目录中的所有 .pt 文件
            for file_name in os.listdir(custom_voices_dir):
                if file_name.endswith('.pt'):
                    spk_name = file_name[:-3]  # 去掉 .pt 后缀

                    # 检查是否已经注册
                    if spk_name not in current_choices:
                        # 新文件，需要注册
                        voice_path = os.path.join(custom_voices_dir, file_name)
                        try:
                            # 加载音色文件
                            custom_voice_info = torch.load(voice_path, map_location='cpu')

                            # 注册到系统中
                            if hasattr(cosyvoice.frontend, 'spk2info'):
                                # 创建音色信息结构
                                model_input = {
                                    'embedding': custom_voice_info.get('embedding'),
                                    'llm_embedding': custom_voice_info.get('embedding'),
                                    'sample_rate': custom_voice_info.get('sample_rate', 16000),
                                    'speaker_name': spk_name
                                }

                                # 添加到spk2info字典
                                cosyvoice.frontend.spk2info[spk_name] = model_input
                                logging.info(f"自动注册新音色: {spk_name}")

                        except Exception as e:
                            logging.warning(f"注册音色 {spk_name} 时出错: {e}")

            # 保存更新后的spk2info
            cosyvoice.save_spkinfo()

        # 重新获取更新后的音色列表
        choices = cosyvoice.list_available_spks()

        # 额外检查：确保自定义音色文件存在，如果文件不存在但还在列表中，则过滤掉
        if hasattr(cosyvoice.frontend, 'spk2info'):
            valid_choices = []

            for spk in choices:
                # 检查是否为自定义音色（通过检查文件是否存在）
                voice_path = os.path.join(custom_voices_dir, f"{spk}.pt")
                if os.path.exists(voice_path) or spk not in cosyvoice.frontend.spk2info:
                    # 文件存在，或者是预训练音色，保留
                    valid_choices.append(spk)
                else:
                    # 文件不存在但还在spk2info中，需要清理
                    logging.warning(f"音色 '{spk}' 的文件不存在，将从列表中移除")
                    if spk in cosyvoice.frontend.spk2info:
                        del cosyvoice.frontend.spk2info[spk]

            choices = valid_choices
            # 保存清理后的spk2info
            cosyvoice.save_spkinfo()

        if not choices:
            choices = ['']

        return {"choices": choices, "__type__": "update"}

    except Exception as e:
        logging.error(f"刷新音色列表时出错: {e}")
        return {"choices": [''], "__type__": "update"}

def delete_custom_spk(selected_spk):
    """删除选中的自定义音色"""
    try:
        if not selected_spk or selected_spk == '':
            return "❌ 请选择要删除的音色"

        # 检查是否为自定义音色（检查是否存在对应的文件）
        custom_voices_dir = os.path.join(cosyvoice.model_dir, 'custom_voices')
        voice_path = os.path.join(custom_voices_dir, f"{selected_spk}.pt")

        if not os.path.exists(voice_path):
            return "❌ 音色不存在或不是自定义音色"

        # 从文件系统中删除音色文件
        os.remove(voice_path)

        # 从spk2info字典中删除
        if hasattr(cosyvoice.frontend, 'spk2info') and selected_spk in cosyvoice.frontend.spk2info:
            del cosyvoice.frontend.spk2info[selected_spk]
            # 保存更新后的spk2info
            cosyvoice.save_spkinfo()

        # 强制刷新音色列表，确保注册信息同步更新
        # 重新加载音色列表，确保删除操作生效
        try:
            # 调用模型的音色列表刷新方法
            if hasattr(cosyvoice, 'refresh_spk_list'):
                cosyvoice.refresh_spk_list()

            # 如果模型有重新加载音色信息的方法，调用它
            if hasattr(cosyvoice.frontend, 'load_spkinfo'):
                cosyvoice.frontend.load_spkinfo()
        except Exception as e:
            logging.warning(f"刷新音色列表时出现警告: {e}")

        return f"✅ 音色 '{selected_spk}' 删除成功，注册信息已同步更新"

    except Exception as e:
        return f"❌ 删除失败: {str(e)}"


# ===================== 自定义音色保存功能 =====================
def save_custom_spk(spk_name, prompt_wav_upload, prompt_wav_record):
    """保存自定义音色"""
    try:
        if not spk_name:
            raise ValueError("请输入音色名称")

        # 获取用户提供的音频
        prompt_wav = prompt_wav_upload or prompt_wav_record
        if not prompt_wav:
            raise ValueError("请提供音频样本")

        # 验证音频采样率
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            raise ValueError(f"音频采样率低于{prompt_sr}Hz，请提供更高质量的音频")

        # 创建保存目录
        CUSTOM_VOICES_DIR = os.path.join(cosyvoice.model_dir, 'custom_voices')
        os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)

        # 提取说话人嵌入向量
        embedding = cosyvoice.frontend._extract_spk_embedding(prompt_wav)

        # 保存自定义音色信息
        custom_voice_info = {
            'speaker_name': spk_name,
            'embedding': embedding.cpu(),
            'sample_rate': prompt_sr,
            'model_version': cosyvoice.__class__.__name__
        }

        save_path = os.path.join(CUSTOM_VOICES_DIR, f"{spk_name}.pt")
        torch.save(custom_voice_info, save_path)

        # 更新spk2info字典和文件
        if hasattr(cosyvoice.frontend, 'spk2info'):
            # 提取与3秒极速复刻相同的特征信息
            model_input = cosyvoice.frontend.frontend_zero_shot('', '', prompt_wav, prompt_sr, '')
            del model_input['text']
            del model_input['text_len']
            # 添加embedding键，兼容frontend_sft方法
            model_input['embedding'] = model_input['llm_embedding']
            cosyvoice.frontend.spk2info[spk_name] = model_input
            cosyvoice.save_spkinfo()

        return f"✅ 音色 '{spk_name}' 保存成功！路径：{save_path}"

    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if speed is None:
        gr.Warning('速度参数不能为空，请设置一个有效的速度值（0.5-2.0）')
        yield (cosyvoice.sample_rate, default_data)
        return

    # 验证流式推理模式下速度调节的限制
    if stream and speed != 1.0:
        gr.Warning('流式推理模式下不支持速度调节，速度将自动设置为1.0')
        effective_speed = 1.0
    else:
        effective_speed = speed

    # 验证速度值范围
    if effective_speed < 0.5 or effective_speed > 2.0:
        gr.Warning('速度值必须在0.5到2.0之间，当前值将被限制在有效范围内')
        effective_speed = max(0.5, min(2.0, effective_speed))

    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

# if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
# 保持原代码的注释风格，提醒跨语种模式注意事项（适配CosyVoice3）
# if cross_lingual mode, please make sure that model is Fun-CosyVoice3-0.5B and tts_text is different language from prompt audio
    if mode_checkbox_group in ['自然语言控制']:
        # 校验1：CosyVoice3 指令文本格式补全（核心：固定前缀拼接，符合模型规范）
        # 移除原V1的cosyvoice.instruct判断，替换为CosyVoice3专属的指令格式补全
        if instruct_text.strip() != '':  # 仅当指令非空时补全格式
            instruct_text = 'You are a helpful assistant. ' + instruct_text.strip() + '。<|endofprompt|>'
        else:
            # 指令为空时，给出警告并返回默认数据（保持原代码yield逻辑）
            gr.Warning('您正在使用自然语言控制模式（CosyVoice3）, 请输入有效的instruct文本（如：用广东话朗读）')
            yield (cosyvoice.sample_rate, default_data)

        # 校验2：指令文本非空校验（保持原代码逻辑，优化提示语适配CosyVoice3）
        if instruct_text.strip() == '':
            gr.Warning('您正在使用自然语言控制模式（CosyVoice3）, 请输入instruct文本')
            yield (cosyvoice.sample_rate, default_data)

        # 校验3：提示用户有效参数（反转原V1逻辑，强调prompt音频必需，prompt文本忽略）
        if prompt_wav is None:
            gr.Warning('您正在使用自然语言控制模式（CosyVoice3）, 请上传有效的prompt参考音频（提取音色）')
            yield (cosyvoice.sample_rate, default_data)
        elif prompt_text != '':
            gr.Info('您正在使用自然语言控制模式（CosyVoice3）, prompt文本会被忽略，仅保留prompt音频用于提取音色')
    if mode_checkbox_group in ['跨语种复刻']:
        # 校验1：移除原V1的cosyvoice.instruct判断（CosyVoice3 原生支持跨语种，无该属性）
        # 直接跳过模型兼容性判断，因为Fun-CosyVoice3-0.5B原生支持跨语种复刻

        # 校验2：instruct文本忽略提示（保持原V1逻辑，优化提示语适配CosyVoice3）
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式（CosyVoice3）, instruct文本会被忽略')

        # 校验3：prompt音频非空校验（保持原V1逻辑，优化提示语和格式严谨性）
        if prompt_wav is None or not os.path.exists(prompt_wav):
            gr.Warning('您正在使用跨语种复刻模式（CosyVoice3）, 请提供有效的prompt参考音频（.wav格式，采样率≥16kHz）')
            yield (cosyvoice.sample_rate, default_data)

        # 校验4：跨语种提醒（保持原V1逻辑，优化提示语适配CosyVoice3的特性）
        gr.Info('您正在使用跨语种复刻模式（CosyVoice3）, 请确保合成文本和prompt音频为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        # ===================== 修复：将边界标记添加到tts_text中 =====================
        tts_text = 'You are a helpful assistant.<|endofprompt|>' + tts_text

        # 原有逻辑：保留“参数被忽略”的信息提示
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')

        # 原有逻辑：保留“无可用预训练音色”的警告与返回
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            yield (cosyvoice.sample_rate, default_data)

    # 后续 3s 极速复刻等其他模式逻辑...
    yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data)
        if 'CosyVoice3' in args.model_dir:
            prompt_text = 'You are a helpful assistant.<|endofprompt|>' + prompt_text
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    # Convert seed to integer to fix the numpy random seed issue
    seed_int = int(seed) if seed is not None else None

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed_int)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=effective_speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed_int)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=effective_speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request（CosyVoice3 跨语种增强版）')
        set_all_random_seed(seed_int)
        # 核心：调用 CosyVoice3 兼容的 inference_cross_lingual 方法，保持原参数格式不变
        COSYVOICE3_CROSS_LINGUAL_PREFIX = "You are a helpful assistant.<|endofprompt|>"
        # 给目标文本拼接固定前缀，对齐官方格式
        tts_text = COSYVOICE3_CROSS_LINGUAL_PREFIX + tts_text.strip()
        for i in cosyvoice.inference_cross_lingual(
            tts_text,                # 待合成的跨语种核心文本（与V1一致）
            prompt_wav,              # 必需：参考音频（提取音色+源语言，与V1一致）
            stream=stream,           # 是否流式推理（与V1一致，兼容布尔值）
            speed=effective_speed    # 语音速度系数（与V1一致，已校验0.5-2.0有效值）
        ):
            # 保持原V1的返回格式，兼容前端Gradio音频组件，无任何修改
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct2 inference request（CosyVoice3 增强版自然语言控制）')
        set_all_random_seed(seed_int)

        # 核心：调用 CosyVoice3 专属的 inference_instruct2 方法（替换原V1的 inference_instruct）
        # 参数适配：移除 sft_dropdown，新增 prompt_wav，保留其他兼容参数
        for i in cosyvoice.inference_instruct2(
            tts_text,                # 待合成的核心文本（与V1一致，保持纯净无指令）
            instruct_text,           # 已补全固定前缀的控制指令（CosyVoice3 规范格式）
            prompt_wav,              # 必需：参考音频（用于提取目标说话人音色，替换V1的 sft_dropdown）
            stream=stream,           # 是否流式推理（与V1一致，兼容布尔值）
            speed=effective_speed    # 语音速度系数（与V1一致，已校验0.5-2.0有效值）
        ):
            # 保持原V1的返回格式，兼容前端Gradio音频组件
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())

def recognize_prompt_wav(prompt_wav):
    try:
        if not prompt_wav:
            return ""

        # FunASR 1.2.9的正确参数
        asr_res = asr_model.generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
            # 移除不支持的use_punc参数
            batch_size_s=30
        )

        # 处理结果
        if asr_res and len(asr_res) > 0:
            result_text = asr_res[0]["text"]
            print(f"原始识别结果: {result_text}")  # 调试输出

            # 检查结果是否包含标点
            if "|>" in result_text:
                text_with_punct = result_text.split('|>')[-1]
            else:
                text_with_punct = result_text

            print(f"处理后文本: {text_with_punct}")  # 调试输出
            return text_with_punct
        return "识别失败：未返回结果"
    except Exception as e:
        return f"识别失败：{str(e)}"

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")

        with gr.Row():
            # 左侧控件组
            with gr.Column(scale=1):
                mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            with gr.Column(scale=1):
                instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]])
            with gr.Row():
                stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1], scale=1)
                speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1, scale=1)

            # 随机种子控制
            with gr.Column(scale=1):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")


        gr.Markdown("**自定义音色管理**")
        with gr.Row():
            with gr.Column(scale=1):
                # 预训练音色选择
                sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择音色', value=sft_spk[0])

                # 音色管理按钮组
                with gr.Row():
                    refresh_button = gr.Button("刷新音色", scale=1)
                    delete_spk_button = gr.Button("删除选中音色", scale=1)

        with gr.Row():
            spk_name = gr.Textbox(label="输入自定义音色名称", placeholder="请输入音色名称", value='', scale=1)
            save_spk_status = gr.Textbox(label="操作状态", interactive=False)
        with gr.Row():
            save_spk_button = gr.Button("保存自定义音色", scale=1)

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources=['upload'],  # 纯上传组件，无录音功能
                type='filepath',
                label='选择prompt音频文件，注意采样率不低于16khz',
                scale=1
            )
            # 核心修改：用 gr.Microphone 替代 gr.Audio，实现纯录音（Gradio 3.x 兼容）
            prompt_wav_record = gr.Microphone(
                type='filepath',  # 录制完成后返回临时文件路径，与原项目逻辑完全兼容
                label='录制prompt音频文件，点击麦克风图标开始录音',
                scale=1
            )

        # 文本输入区域
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        # 生成按钮和输出区域
        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        # 绑定事件
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])

        # 自定义音色管理事件
        save_spk_button.click(save_custom_spk,
                              inputs=[spk_name, prompt_wav_upload, prompt_wav_record],
                              outputs=[save_spk_status])
        refresh_button.click(refresh_sft_spk, inputs=[], outputs=[sft_dropdown])
        delete_spk_button.click(delete_custom_spk, inputs=[sft_dropdown], outputs=[save_spk_status])

        # 绑定音频上传和录制的识别事件
        prompt_wav_upload.change(
            fn=recognize_prompt_wav,  # 使用正确的识别函数
            inputs=[prompt_wav_upload],
            outputs=[prompt_text]
        )
        prompt_wav_record.change(
            fn=recognize_prompt_wav,  # 使用正确的识别函数
            inputs=[prompt_wav_record],
            outputs=[prompt_text]
        )


    demo.queue(max_size=4)
    print("\n" + "="*50)
    print(f"🔗 本地访问地址: \033[1;32mhttp://localhost:8000\033[0m")  # 绿色高亮
    print("="*50 + "\n")
    demo.launch(
        server_name="localhost",  # 强制绑定 localhost
        server_port=8000,        # 端口号（可选，默认8000）
        share=False              # 无需公共链接时设为False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice3-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()