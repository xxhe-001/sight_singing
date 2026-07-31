# midi_generator.py
import os
import pretty_midi

class MidiGenerator:
    """简易 MIDI 旋律生成器（支持简谱文本）"""
    
    SCALE_MAP = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

    def __init__(self, tempo=100, base_octave=4):
        self.tempo = tempo
        self.base_octave = base_octave

    def from_jianpu(self, text, out_path):
        beat_dur = 60.0 / self.tempo
        tokens = text.replace(',', ' ').replace('|', ' ').split()
        notes = []
        for token in tokens:
            if not token:
                continue
            if '/' in token:
                pitch_part, dur_part = token.split('/', 1)
            else:
                pitch_part, dur_part = token, '4'
            try:
                dur_fraction = float(dur_part)
                duration = beat_dur * (4.0 / dur_fraction)
            except:
                duration = beat_dur

            if pitch_part == '0':
                notes.append((-1, duration))
                continue

            octave_shift = 0
            while pitch_part.endswith('`'):
                octave_shift -= 1
                pitch_part = pitch_part[:-1]
            while pitch_part.endswith('^'):
                octave_shift += 1
                pitch_part = pitch_part[:-1]

            accidental = 0
            if '#' in pitch_part:
                accidental = 1
                pitch_part = pitch_part.replace('#', '')
            elif 'b' in pitch_part:
                accidental = -1
                pitch_part = pitch_part.replace('b', '')

            degree = int(pitch_part)
            if degree < 1 or degree > 7:
                raise ValueError(f"音符 {degree} 超出 1-7 范围: {token}")

            # 修正：base_octave + 1 才能匹配标准 MIDI 音高
            midi_num = 12 * (self.base_octave + 1 + octave_shift) + self.SCALE_MAP[degree] + accidental
            notes.append((midi_num, duration))

        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        track = pretty_midi.Instrument(program=0)
        start = 0.0
        for pitch, dur in notes:
            if pitch == -1:
                start += dur
                continue
            track.notes.append(pretty_midi.Note(
                velocity=100, pitch=pitch,
                start=start, end=start + dur
            ))
            start += dur
        midi.instruments.append(track)
        midi.write(out_path)
        return out_path

# ========== 20 首视唱练习曲 ==========
EXAM_PIECES = [
    # 原来 03~20 依次改为 01~18
    {"name": "01-音阶上下行", "tempo": 90, "jianpu": """
        1/4 2/4 3/4 4/4 5/4 6/4 7/4 1^/4 |
        1^/4 7/4 6/4 5/4 4/4 3/4 2/4 1/2
    """},
    {"name": "02-三度模进", "tempo": 90, "jianpu": """
        1/4 3/4 2/4 4/4 3/4 5/4 4/4 6/4 |
        5/4 7/4 6/4 1^/4 7/4 2^/4 1^/2
    """},
    {"name": "03-附点节奏", "tempo": 80, "jianpu": """
        1/4 2/8 3/8 2/8 1/8 2/4. 3/8 |
        4/4 3/8 2/8 3/4 2/4. 1/8 2/2
    """},
    {"name": "04-切分音", "tempo": 80, "jianpu": """
        1/8 3/4 3/8 1/8 2/4 2/8 |
        3/8 5/4 5/8 3/8 2/4 2/8 1/2
    """},
    {"name": "05-四度跳进", "tempo": 90, "jianpu": """
        1/4 4/4 2/4 5/4 3/4 6/4 4/4 7/4 |
        5/4 1^/4 6/4 2^/4 7/4 3^/4 1^/2
    """},
    {"name": "06-五度跳进", "tempo": 90, "jianpu": """
        1/4 5/4 2/4 6/4 3/4 7/4 4/4 1^/4 |
        5/4 2^/4 6/4 3^/4 7/4 4^/4 1^/2
    """},
    {"name": "07-低音练习", "tempo": 85, "jianpu": """
        5`/4 6`/4 7`/4 1/4 2/4 3/4 4/4 5/4 |
        4/4 3/4 2/4 1/4 7`/4 6`/4 5`/2
    """},
    {"name": "08-高音练习", "tempo": 85, "jianpu": """
        1^/4 7/4 6/4 5/4 4/4 3/4 2/4 1/4 |
        5/4 6/4 7/4 1^/4 2^/4 1^/4 7/2
    """},
    {"name": "09-综合节奏", "tempo": 100, "jianpu": """
        3/8 3/8 4/8 5/4 3/8 2/4 1/8 3/8 2/2 |
        1/8 1/8 2/8 3/4 2/8 3/8 4/8 5/4 3/8 2/4 1/2
    """},
    {"name": "10-大调旋律", "tempo": 95, "jianpu": """
        5/4 1^/4 7/4 6/4 5/4 6/4 5/4 3/4 |
        2/4 3/4 5/4 6/4 5/4 3/4 2/4 1/2
    """},
    {"name": "11-小调色彩", "tempo": 95, "jianpu": """
        6/4 1^/4 7/4 6/4 5/4 6/4 5/4 3/4 |
        2/4 3/4 1/4 6`/4 7`/4 1/4 2/4 3/4 6/2
    """},
    {"name": "12-连音线", "tempo": 85, "jianpu": """
        1/4 2/8 3/8 4/4. 3/8 2/4. 1/8 |
        5/4 6/8 5/8 4/4 3/4 2/8 3/8 1/2
    """},
    {"name": "13-三连音", "tempo": 120, "jianpu": """
        1/3 2/3 3/3 4/4 5/4 6/4 |
        7/4 1^/3 7/3 6/3 5/4 4/4 3/4 2/4 1/2
    """},
    {"name": "14-装饰音", "tempo": 80, "jianpu": """
        1/16 2/16 3/8 4/4 3/8 5/16 4/16 3/8 2/4 |
        1/4 3/16 2/16 1/8 6`/8 5`/8 1/2
    """},
    {"name": "15-音程跳进", "tempo": 90, "jianpu": """
        1/4 6/4 2/4 7/4 3/4 1^/4 4/4 2^/4 |
        5/4 3^/4 6/4 4^/4 7/4 5^/4 1^/2
    """},
    {"name": "16-长音保持", "tempo": 70, "jianpu": """
        1/2 3/2 5/2 3/2 | 1/4 2/4 3/4 5/4 6/4 5/2 |
        6/2 5/2 3/2 1/2 | 2/4 3/4 5/4 6/4 1^/2
    """},
    {"name": "17-混合拍子", "tempo": 110, "jianpu": """
        1/4 2/4 3/4 5/8 6/8 5/8 3/4 |
        1/8 2/8 3/8 5/8 6/8 5/8 3/8 2/8 1/4 2/2
    """},
    {"name": "18-综合练习", "tempo": 100, "jianpu": """
        5/4 3/4 1/4 2/4 3/4 1/4 6`/4 5`/4 |
        6`/4 1/4 2/4 3/4 5/4 3/4 2/4 1/2 |
        2/4 3/4 5/4 6/4 1^/4 7/4 6/4 5/4 |
        3/4 5/4 6/4 5/4 3/4 2/4 1/4 5`/4 1/2
    """},
    # 以下两首放在最后，编号 19、20
    {"name": "19-小星星", "tempo": 100, "jianpu": """
        1/4 1/4 5/4 5/4 6/4 6/4 5/2 |
        4/4 4/4 3/4 3/4 2/4 2/4 1/2 |
        5/4 5/4 4/4 4/4 3/4 3/4 2/2 |
        5/4 5/4 4/4 4/4 3/4 3/4 2/2 |
        1/4 1/4 5/4 5/4 6/4 6/4 5/2 |
        4/4 4/4 3/4 3/4 2/4 2/4 1/2
    """},
    {"name": "20-浏阳河", "tempo": 100, "jianpu": """
        5/8 6/8 1/4 6/8 5/8 3/8 5/4 |
        6/8 5/8 3/8 5/8 1/8 2/8 3/4 |
        5/8 3/8 2/8 1/8 6/8 5/8 3/8 5/4 |
        6/4 1/8 6/8 5/8 3/8 5/8 6/4 |
        1/8 6/8 5/8 3/8 5/8 6/8 5/4 |
        1/8 2/8 3/8 5/8 2/8 1/8 6/4 |
        5/8 6/8 1/4 6/8 5/8 3/8 5/4 |
        6/8 5/8 3/8 5/8 1/8 2/8 3/2
    """}
]


def generate_all_exam_midis(scores_dir="test_data/scores"):
    """自动生成所有练习曲的 MIDI 文件（如果不存在）"""
    os.makedirs(scores_dir, exist_ok=True)
    for piece in EXAM_PIECES:
        filename = piece["name"] + ".mid"
        filepath = os.path.join(scores_dir, filename)
        if not os.path.exists(filepath):
            try:
                gen = MidiGenerator(tempo=piece["tempo"])
                gen.from_jianpu(piece["jianpu"], filepath)
                print(f"✅ 已生成曲目: {filename}")
            except Exception as e:
                print(f"❌ 生成 {filename} 失败: {e}")