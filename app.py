# -*- coding: utf-8 -*-
import os
import re
import time
import uuid
import numpy as np
import librosa
import pretty_midi
from fastdtw import fastdtw
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import warnings
# 导入 MIDI 生成器
from midi_generator import generate_all_exam_midis

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

SCORES_DIR = os.path.join("test_data", "scores")
RECORDINGS_DIR = os.path.join("test_data", "recordings")
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ============================
# 生成正确时值的"小星星"MIDI
# ============================
def generate_sample_midi():
    midi_path = os.path.join(SCORES_DIR, "小星星.mid")
    if os.path.exists(midi_path):
        return midi_path
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    melody_with_duration = [
        (60, 0.5), (60, 0.5), (67, 0.5), (67, 0.5), (69, 0.5), (69, 0.5), (67, 1.0),
        (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 0.5), (62, 0.5), (60, 1.0),
        (67, 0.5), (67, 0.5), (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 1.0),
        (67, 0.5), (67, 0.5), (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 1.0),
        (60, 0.5), (60, 0.5), (67, 0.5), (67, 0.5), (69, 0.5), (69, 0.5), (67, 1.0),
        (65, 0.5), (65, 0.5), (64, 0.5), (64, 0.5), (62, 0.5), (62, 0.5), (60, 1.0),
    ]
    start = 0.0
    for pitch, dur in melody_with_duration:
        end = start + dur
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        piano.notes.append(note)
        start = end
    midi.instruments.append(piano)
    midi.write(midi_path)
    return midi_path

def generate_liuyanghe_midi():
    """生成浏阳河旋律的MIDI文件"""
    midi_path = os.path.join(SCORES_DIR, "浏阳河.mid")
    if os.path.exists(midi_path):
        return midi_path
    
    # 创建MIDI对象，设定速度100 BPM
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    piano = pretty_midi.Instrument(program=0)  # 钢琴音色
    
    # 浏阳河主旋律 (C调，2/4拍)
    # 音符：(MIDI音高, 时长_秒)
    # 简谱音高对应：1=C4(60), 2=D4(62), 3=E4(64), 5=G4(67), 6=A4(69)
    # 节奏：四分音符=0.6秒(100BPM)，八分音符=0.3秒
    melody = [
        # 第一句：5 6 1 6 5 3 5 | 6 5 3 5 1 2 3 |
        (67, 0.3), (69, 0.3), (60, 0.6), (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.6),
        (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.3), (60, 0.3), (62, 0.3), (64, 0.6),
        
        # 第二句：5 3 2 1 6 5 3 5 | 6 1 6 5 3 5 6 |
        (67, 0.3), (64, 0.3), (62, 0.3), (60, 0.3), (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.6),
        (69, 0.6), (60, 0.3), (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.3), (69, 0.6),
        
        # 第三句：1 6 5 3 5 6 5 | 1 2 3 5 2 1 6 |
        (60, 0.3), (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.3), (69, 0.3), (67, 0.6),
        (60, 0.3), (62, 0.3), (64, 0.3), (67, 0.3), (62, 0.3), (60, 0.3), (69, 0.6),
        
        # 第四句：5 6 1 6 5 3 5 | 6 5 3 5 1 2 3 |
        (67, 0.3), (69, 0.3), (60, 0.6), (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.6),
        (69, 0.3), (67, 0.3), (64, 0.3), (67, 0.3), (60, 0.3), (62, 0.3), (64, 1.2),  # 最后长音
    ]
    
    # 将音符添加到乐器轨道
    start_time = 0.0
    for pitch, duration in melody:
        end_time = start_time + duration
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
        piano.notes.append(note)
        start_time = end_time
    
    midi.instruments.append(piano)
    midi.write(midi_path)
    return midi_path
# ============================
# 评估引擎（极度放宽要求）
# ============================
def hz_to_midi(f0):
    midi = librosa.hz_to_midi(f0)
    return np.where(np.isfinite(midi), midi, -1)

class SingingEvaluator:
    def __init__(self, audio_path, midi_path, sr=22050, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length
        self.frame_time = hop_length / sr
        self.audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        self.midi_data = pretty_midi.PrettyMIDI(midi_path)
        self.ref_notes = self._extract_midi_notes()
        self.total_duration = max([n.end for n in self.ref_notes]) if self.ref_notes else 0
        self.ref_pitch_seq = self._build_reference_pitch_sequence()
        self.user_pitch_seq, self.voiced_flags = self._extract_pitch()
        self.alignment_path = self._align_sequences()

    def _extract_midi_notes(self):
        notes = []
        for inst in self.midi_data.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        if not notes:
            for inst in self.midi_data.instruments:
                notes.extend(inst.notes)
        notes.sort(key=lambda n: n.start)
        return notes

    def _build_reference_pitch_sequence(self):
        if not self.ref_notes:
            return np.array([])
        total_frames = int(np.ceil(self.total_duration / self.frame_time))
        ref = np.full(total_frames, -1, dtype=np.float32)
        for note in self.ref_notes:
            start_frame = int(note.start / self.frame_time)
            end_frame = int(note.end / self.frame_time)
            if end_frame > total_frames:
                end_frame = total_frames
            ref[start_frame:end_frame] = note.pitch
        return ref

    def _extract_pitch(self):
        f0, voiced_flag, _ = librosa.pyin(
            self.audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        user_midi = hz_to_midi(f0)
        user_midi[~voiced_flag] = -1
        return user_midi, voiced_flag

    def _distance_func(self, user_frame, ref_frame):
        user_pitch = user_frame
        ref_pitch = ref_frame
        if ref_pitch == -1 and user_pitch == -1:
            return 0.0
        if ref_pitch == -1 and user_pitch != -1:
            return 10.0
        if ref_pitch != -1 and user_pitch == -1:
            return 10.0
        return abs(float(user_pitch) - float(ref_pitch))

    def _align_sequences(self):
        if len(self.ref_pitch_seq) == 0 or len(self.user_pitch_seq) == 0:
            return []
        distance, path = fastdtw(
            self.user_pitch_seq.reshape(-1, 1),
            self.ref_pitch_seq.reshape(-1, 1),
            radius=30,
            dist=lambda u, r: self._distance_func(u[0], r[0])
        )
        return path

    def evaluate(self):
        results = {
            'note_results': [],
            'pitch_score': 0.0,
            'rhythm_score': 0.0,
            'overall_score': 0.0,
            'suggestions': []
        }
        if not self.ref_notes or len(self.alignment_path) == 0:
            results['suggestions'].append("未能对齐，请检查录音与乐谱是否对应。")
            return results

        ref_to_user_frames = {}
        for u_idx, r_idx in self.alignment_path:
            ref_to_user_frames.setdefault(r_idx, []).append(u_idx)

        note_pitch_scores = []
        note_rhythm_scores = []
        note_details = []

        for note in self.ref_notes:
            ref_start_frame = int(note.start / self.frame_time)
            ref_end_frame = int(note.end / self.frame_time)
            user_frames = []
            for r in range(ref_start_frame, min(ref_end_frame, len(self.ref_pitch_seq))):
                if r in ref_to_user_frames:
                    user_frames.extend(ref_to_user_frames[r])
            
            if not user_frames:
                note_details.append({
                    'start': note.start,
                    'end': note.end,
                    'pitch_name': pretty_midi.note_number_to_name(note.pitch),
                    'midi_note': note.pitch,
                    'pitch_dev_cents': None,
                    'timing_offset': None,
                    'is_missed': True,
                    'error_type': 'missed'  # 漏唱才标红
                })
                note_pitch_scores.append(0)
                note_rhythm_scores.append(0)
                continue

            frame_pitches = []
            for uf in user_frames:
                if uf < len(self.user_pitch_seq):
                    p = self.user_pitch_seq[uf]
                    if p != -1:
                        frame_pitches.append(p)
            
            if frame_pitches:
                mean_user_pitch = np.mean(frame_pitches)
                dev_cents = (mean_user_pitch - note.pitch) * 100
                # 极度宽松的音准评分：允许很大偏差
                pitch_score = max(0, 100 - abs(dev_cents) * 0.05)  # 非常宽松
            else:
                dev_cents = None
                pitch_score = 0

            user_start_times = []
            for uf in user_frames:
                if uf < len(self.user_pitch_seq) and self.user_pitch_seq[uf] != -1:
                    user_start_times.append(uf * self.frame_time)
                    break
            
            if user_start_times:
                actual_start = user_start_times[0]
                timing_offset = actual_start - note.start
                # 极度宽松的节奏评分：允许很大的时间偏差
                rhythm_score = max(0, 100 - abs(timing_offset) * 1000 * 0.005)  # 非常宽松
            else:
                timing_offset = None
                rhythm_score = 0

            # 判断错误类型 - 极度宽松的阈值
            error_type = 'good'
            
            # 音准判断：只有差了一个半音以上才标红
            if dev_cents is not None:
                abs_dev = abs(dev_cents)
                if abs_dev > 200:  # 差2个半音才标红（原来80）
                    error_type = 'pitch_error'
                elif abs_dev > 100:  # 差1个半音标黄（原来40）
                    error_type = 'pitch_slight'
            
            # 节奏判断：只有差了很多才标记
            if timing_offset is not None:
                abs_offset = abs(timing_offset)
                if abs_offset > 1.0:  # 差1秒以上才标红（原来0.5）
                    if error_type == 'pitch_error':
                        error_type = 'both_error'
                    elif error_type != 'pitch_error':
                        error_type = 'rhythm_error'
                elif abs_offset > 0.5:  # 差0.5秒标黄（原来0.3）
                    if error_type == 'good':
                        error_type = 'rhythm_slight'

            note_details.append({
                'start': note.start,
                'end': note.end,
                'pitch_name': pretty_midi.note_number_to_name(note.pitch),
                'midi_note': note.pitch,
                'pitch_dev_cents': round(dev_cents) if dev_cents is not None else None,
                'timing_offset': round(timing_offset, 3) if timing_offset is not None else None,
                'is_missed': False,
                'error_type': error_type
            })
            note_pitch_scores.append(pitch_score)
            note_rhythm_scores.append(rhythm_score)

        if note_pitch_scores:
            results['pitch_score'] = round(np.mean(note_pitch_scores), 1)
            results['rhythm_score'] = round(np.mean(note_rhythm_scores), 1)
            results['overall_score'] = round(0.6 * results['pitch_score'] + 0.4 * results['rhythm_score'], 1)
        results['note_results'] = note_details

        # 只在有严重问题时才给建议
        pitch_devs = [nd['pitch_dev_cents'] for nd in note_details if nd['pitch_dev_cents'] is not None]
        if pitch_devs:
            avg_pitch_dev = np.mean(pitch_devs)
            if abs(avg_pitch_dev) > 100:  # 整体偏差超过1个半音才提示
                direction = "偏高" if avg_pitch_dev > 0 else "偏低"
                results['suggestions'].append(f"整体音高{direction}约{abs(avg_pitch_dev):.0f}音分。")

        timing_offsets = [nd['timing_offset'] for nd in note_details if nd['timing_offset'] is not None]
        if timing_offsets:
            avg_offset = np.mean(timing_offsets)
            if abs(avg_offset) > 0.8:  # 整体节奏偏差超过0.8秒才提示
                direction = "偏慢" if avg_offset > 0 else "偏快"
                results['suggestions'].append(f"整体节奏{direction}约{abs(avg_offset)*1000:.0f}毫秒。")

        # 只在有严重错误时才给具体音符建议
        for nd in note_details:
            if nd.get('is_missed'):
                results['suggestions'].append(f"音符 {nd['pitch_name']} (时间 {nd['start']:.1f}s) 未检测到。")
            elif nd['error_type'] in ['pitch_error', 'both_error']:
                d = "偏高" if nd['pitch_dev_cents'] > 0 else "偏低"
                results['suggestions'].append(f"音符 {nd['pitch_name']} (时间 {nd['start']:.1f}s) 音高{d}较多。")
            elif nd['error_type'] == 'rhythm_error':
                results['suggestions'].append(f"音符 {nd['pitch_name']} (时间 {nd['start']:.1f}s) 节奏偏差较大。")
        
        return results

# ============================
# ABC 生成辅助函数
# ============================

def midi_to_abc_pitch(midi_num):
    """
    ABC 记谱法音高规则（L:1/4 基准）：
      C,  = C3  (MIDI 48)
      C   = C4  中央C (MIDI 60)
      c   = C5  高音C (MIDI 72)
      c'  = C6  (MIDI 84)
    """
    note_names = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']
    name = note_names[midi_num % 12]
    octave = midi_num // 12 - 1

    if octave < 4:
        return name + "," * (4 - octave)
    elif octave == 4:
        return name
    elif octave == 5:
        return name.lower()
    else:
        return name.lower() + "'" * (octave - 5)

def duration_to_abc_length(duration_sec, tempo):
    quarter_dur = 60.0 / tempo
    q = duration_sec / quarter_dur

    if abs(q - 4.0) < 0.2:
        return "4", 4.0
    elif abs(q - 2.0) < 0.15:
        return "2", 2.0
    elif abs(q - 1.0) < 0.1:
        return "", 1.0
    elif abs(q - 0.5) < 0.08:
        return "/2", 0.5
    elif abs(q - 0.25) < 0.05:
        return "/4", 0.25
    else:
        return "", 1.0

def split_duration_to_abc_parts(total_dur_sec, tempo):
    quarter_dur = 60.0 / tempo
    q = total_dur_sec / quarter_dur
    parts = []

    while q > 0.001:
        if q >= 4.0:
            parts.append(("4", 4.0))
            q -= 4.0
        elif q >= 2.0:
            parts.append(("2", 2.0))
            q -= 2.0
        elif q >= 1.0:
            parts.append(("", 1.0))
            q -= 1.0
        elif q >= 0.5:
            parts.append(("/2", 0.5))
            q -= 0.5
        else:
            parts.append(("/4", 0.25))
            q -= 0.25
    return parts

# ============================
# Flask 路由
# ============================
@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('static', 'service-worker.js',
                               mimetype='application/javascript')

@app.route('/')
def index():
    return render_template('index.html')

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

@app.route('/api/scores')
def list_scores():
    generate_all_exam_midis(SCORES_DIR)
    files = [f for f in os.listdir(SCORES_DIR) if f.lower().endswith('.mid')]
    files.sort(key=natural_key)
    return jsonify([{'name': os.path.splitext(f)[0], 'filename': f} for f in files])

@app.route('/api/score/<filename>/abc')
def get_abc(filename):
    safe_path = os.path.join(SCORES_DIR, filename)
    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404

    midi = pretty_midi.PrettyMIDI(safe_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)

    if not notes:
        return jsonify({'error': 'No notes'}), 400

    tempos = midi.get_tempo_changes()
    tempo = tempos[1][0] if len(tempos[0]) > 0 else 120

    tokens = []
    current_beat = 0.0
    bar_duration = 4.0
    prev_end = 0.0

    for note in notes:
        if note.start > prev_end + 0.001:
            rest_dur = note.start - prev_end
            rest_parts = split_duration_to_abc_parts(rest_dur, tempo)
            for abc_len, beats in rest_parts:
                if current_beat + beats > bar_duration + 0.001:
                    tokens.append("|")
                    current_beat = 0.0
                tokens.append(f"z{abc_len}")
                current_beat += beats
                if current_beat >= bar_duration - 0.001:
                    tokens.append("|")
                    current_beat = 0.0

        pitch = midi_to_abc_pitch(note.pitch)
        abc_len, beats = duration_to_abc_length(note.end - note.start, tempo)

        if current_beat + beats > bar_duration + 0.001:
            tokens.append("|")
            current_beat = 0.0

        tokens.append(f"{pitch}{abc_len}")
        current_beat += beats
        if current_beat >= bar_duration - 0.001:
            tokens.append("|")
            current_beat = 0.0

        prev_end = note.end

    if tokens and tokens[-1] != "|":
        tokens.append("|")
    tokens.append("]")

    measures = []
    current_measure = []
    for token in tokens:
        if token == "|":
            if current_measure:
                measures.append(" ".join(current_measure))
                current_measure = []
            measures.append("|")
        elif token == "]":
            if current_measure:
                measures.append(" ".join(current_measure))
                current_measure = []
            measures.append("]")
        else:
            current_measure.append(token)
    if current_measure:
        measures.append(" ".join(current_measure))

    lines = []
    line_measures = []
    for item in measures:
        if item == "]":
            if line_measures:
                lines.append(" | ".join(line_measures))
                line_measures = []
            if lines:
                lines[-1] += " ]"
            else:
                lines.append("]")
        elif item == "|":
            pass
        else:
            line_measures.append(item)
            if len(line_measures) == 4:
                lines.append(" | ".join(line_measures))
                line_measures = []
    if line_measures:
        lines.append(" | ".join(line_measures))

    if lines and not lines[-1].rstrip().endswith("]"):
        lines[-1] += " ]"

    abc_lines = ["X:1", "T:" + os.path.splitext(filename)[0], "M:4/4", f"Q:1/4={int(tempo)}", "L:1/4", "K:C"]
    abc_lines.extend(lines)
    abc_text = "\n".join(abc_lines)
    return abc_text, 200, {'Content-Type': 'text/plain'}

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    midi_filename = request.form.get('midi', '')
    
    if not midi_filename:
        return jsonify({'error': 'Missing midi parameter'}), 400
    
    midi_path = os.path.join(SCORES_DIR, midi_filename)
    if not os.path.exists(midi_path):
        return jsonify({'error': 'MIDI file not found'}), 404

    unique_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_filename = f"recording_{timestamp}_{unique_id}.wav"
    audio_path = os.path.join(RECORDINGS_DIR, audio_filename)
    audio_file.save(audio_path)

    try:
        evaluator = SingingEvaluator(audio_path, midi_path)
        results = evaluator.evaluate()
        results['recording'] = audio_filename
        
        # 删除本地wav录音文件
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已删除录音文件: {audio_path}")
        except Exception as e:
            print(f"删除录音文件失败: {e}")
        
        return jsonify(results)
    except Exception as e:
        # 发生错误时也尝试删除录音文件
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)