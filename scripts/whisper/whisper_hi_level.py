import fire
import whisper
import time

def parse_results(result):
    parsed = []
    for seg in result['segments']:
        info = {'text': seg['text'],
                'beg': seg['start'],
                'end': seg['end'],
                'logprob': seg['avg_logprob'],
                'no_speech_prob': seg['no_speech_prob']}
        parsed.append(info)
    return parsed

def main(audio_fid: str):
    beg_mod = time.time()
    model = whisper.load_model("base")
    end_mod = time.time()
    print(f'MODEL LOADING TIME: {end_mod - beg_mod}')

    # load audio and pad/trim it to fit 30 seconds
    beg_pre = time.time()
    audio = whisper.load_audio(audio_fid)
    audio = whisper.pad_or_trim(audio)
    end_pre = time.time()
    print(f'AUDIO PREP TIME: {end_pre - beg_pre}')

    beg_res = time.time()
    result = model.transcribe(audio)
    parsed = parse_results(result)
    end_res = time.time()
    print(f'TRASCRIBE TIME: {end_res - beg_res}')

    print('\n'.join(seg['text'] for seg in parsed))


if __name__ == '__main__':
    fire.Fire(main)