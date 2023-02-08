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

def main(filename: str,
         sample_rate: int = 16_000,
         audio_dur=30):
    model = whisper.load_model("base")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)

    window_size = sample_rate * audio_dur
    num_windows = (len(audio) // window_size) + 1
    transcripts = []

    for i in range(num_windows):
        seg = audio[i*window_size:(i+1)*window_size]
        seg = whisper.pad_or_trim(seg)

        result = model.transcribe(seg)
        parsed = parse_results(result)

        transcripts += parsed

    print('\n'.join(seg['text'] for seg in transcripts))


if __name__ == '__main__':
    fire.Fire(main)