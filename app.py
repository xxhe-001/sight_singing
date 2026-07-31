# -*- coding: utf-8 -*-
import os
import sys
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

# ---------- 确定资源路径 ----------
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

SCORES_DIR = os.path.join(base_dir, "test_data", "scores")
RECORDINGS_DIR = os.path.join(base_dir, "test_data", "recordings")
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ============================
# 全局 JSON 错误处理器
# ============================
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': '请求参数错误'}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '资源未找到'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': '服务器内部错误'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # 将所有未处理异常转为 JSON
    return jsonify({'error': str(e)}), 500

# ============================
# 评估引擎（保持原有逻辑）
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
                    'error_type': 'missed'
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
                pitch_score = max(0, 100 - abs(dev_cents) * 0.05)
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
                rhythm_score = max(0, 100 - abs(timing_offset) * 1000 * 0.005)
            else:
                timing_offset = None
                rhythm_score = 0

            error_type = 'good'
            if dev_cents is not None:
                abs_dev = abs(dev_cents)
                if abs_dev > 200:
                    error_type = 'pitch_error'
                elif abs_dev > 100:
                    error_type = 'pitch_slight'
            
            if timing_offset is not None:
                abs_offset = abs(timing_offset)
                if abs_offset > 1.0:
                    if error_type == 'pitch_error':
                        error_type = 'both_error'
                    elif error_type != 'pitch_error':
                        error_type = 'rhythm_error'
                elif abs_offset > 0.5:
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

        pitch_devs = [nd['pitch_dev_cents'] for nd in note_details if nd['pitch_dev_cents'] is not None]
        if pitch_devs:
            avg_pitch_dev = np.mean(pitch_devs)
            if abs(avg_pitch_dev) > 100:
                direction = "偏高" if avg_pitch_dev > 0 else "偏低"
                results['suggestions'].append(f"整体音高{direction}约{abs(avg_pitch_dev):.0f}音分。")

        timing_offsets = [nd['timing_offset'] for nd in note_details if nd['timing_offset'] is not None]
        if timing_offsets:
            avg_offset = np.mean(timing_offsets)
            if abs(avg_offset) > 0.8:
                direction = "偏慢" if avg_offset > 0 else "偏快"
                results['suggestions'].append(f"整体节奏{direction}约{abs(avg_offset)*1000:.0f}毫秒。")

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
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已删除录音文件: {audio_path}")
        except Exception as e:
            print(f"删除录音文件失败: {e}")
        return jsonify(results)
    except Exception as e:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    generate_all_exam_midis(SCORES_DIR)
    import webbrowser
    import threading
    def open_browser():
        webbrowser.open('http://127.0.0.1:5001')
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, host='127.0.0.1', port=5001)