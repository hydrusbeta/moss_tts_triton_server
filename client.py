import argparse
import base64
import json
import os.path
import subprocess
import tempfile
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema import ValidationError
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

PYTHON_EXECUTABLE = 'python3'
CHECKPOINT_DIR = 'moss_tts_local_clipper_checkpoint'

# From https://huggingface.co/OpenMOSS-Team/MOSS-TTS#supported-languages
SUPPORTED_LANGUAGES = ['zh', 'es', 'it', 'ru', 'pl', 'da', 'el', 'en', 'fr', 'hu', 'fa', 'pt', 'sv', 'tr', 'de', 'ja', 'ko', 'ar', 'cs']

# global variable. 1 second of generated audio for every ~12.5 tokens
MAX_NEW_TOKENS = 10 # todo: change default to 250

app = Flask(__name__)

def register_methods(cache, max_new_tokens):
    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = max_new_tokens if max_new_tokens else 10 # todo: change default to 250

    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            (user_text, character, style_text, language, emotion, emotion_energy, audio_temperature, audio_top_k,
             audio_top_p, repetition_penalty, rvq_codebook_layers, output_filename_sans_extension, gpu_id, session_id) \
                = parse_inputs()
            with tempfile.TemporaryDirectory() as tempdir:
                execute_program(user_text, character, style_text, language, emotion, emotion_energy, audio_temperature,
                                audio_top_k, audio_top_p, repetition_penalty, rvq_codebook_layers, gpu_id, tempdir)
                copy_output(tempdir, output_filename_sans_extension, session_id)
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = ('An error occurred while generating the output: \n' + traceback.format_exc() +
                       '\n\nPayload:\n' + json.dumps(request.json))

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    @app.route('/gpu-info', methods=['GET'])
    def get_gpu_info():
        return hsc.get_gpu_info_from_another_venv(PYTHON_EXECUTABLE)

    def get_emotion_index(emotion, character):
        emotions = get_available_traits(character)
        if emotion in emotions:
            return emotions.index(emotion)
        else:
            raise BadInputException(f'emotion {emotion} is not valid for character {character}. Expected one of {emotions}.')

    @app.route('/available-traits/<character>', methods=['GET'])
    def get_available_traits(character):
        traits = []
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        embedded_styles_path = os.path.join(character_dir, CHECKPOINT_DIR , 'embedded_styles.json')
        if os.path.exists(embedded_styles_path):
            with open(embedded_styles_path, 'r') as file:
                traits = json.load(file)
        else:
            # This is not necessarily an error condition. It just means there are no emotions embedded in the model.
            print(f"{embedded_styles_path} does not exist", flush=True)
        return traits

    schema = {
        'type': 'object',
        'properties': {
            'Inputs': {
                'type': 'object',
                'properties': {
                    'User Text': {'type': 'string'},
                },
                'required': ['User Text']
            },
            'Options': {
                'type': 'object',
                'properties': {
                    'Character': {'type': 'string'},
                    'Style Text': {'type': 'string'},
                    'Language': {'enum': SUPPORTED_LANGUAGES},
                    'Emotion': {'type': 'string'},
                    'Emotion Energy': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'Audio Temperature': {'type': 'number', 'minimum': .00001, 'maximum': 2},
                    'Audio Top-K': {'type': 'integer', 'minimum': 0, 'maximum': 200},
                    'Audio Top-P': {'type': 'number', 'minimum': .00001, 'maximum': 1},
                    'Repetition Penalty': {'type': 'number', 'minimum': .00001, 'maximum': 2},
                    'RVQ Codebook Layers': {'type': 'integer', 'minimum': 1, 'maximum': 32},
                },
                'required': ['Character']
            },
            'Output File': {'type': 'string'},
            'GPU ID': {'type': ['string', 'integer']},
            'Session ID': {'type': ['string', 'null']}
        },
        'required': ['Inputs', 'Options', 'Output File', 'GPU ID', 'Session ID']
    }

    def parse_inputs():
        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        user_text = request.json['Inputs']['User Text']
        character = request.json['Options']['Character']
        style_text = request.json['Options'].get('Style Text')
        language = request.json['Options'].get('Language')
        emotion = request.json['Options'].get('Emotion')
        emotion_energy = request.json['Options'].get('Emotion Energy')
        audio_temperature = request.json['Options'].get('Audio Temperature')
        audio_top_k = request.json['Options'].get('Audio Top-K')
        audio_top_p = request.json['Options'].get('Audio Top-P')
        repetition_penalty = request.json['Options'].get('Repetition Penalty')
        rvq_codebook_layers = request.json['Options'].get('RVQ Codebook Layers')
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']

        return (user_text, character, style_text, language, emotion, emotion_energy, audio_temperature, audio_top_k,
                audio_top_p, repetition_penalty, rvq_codebook_layers, output_filename_sans_extension, gpu_id,
                session_id)

    class BadInputException(Exception):
        pass

    def get_character_index(character):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        characters_json = os.path.join(character_dir, CHECKPOINT_DIR, 'embedded_characters.json')
        if not os.path.exists(characters_json):
            return 0 # Assuming this is not a multispeaker model. Hopefully 0 will just work.
        with open(characters_json, 'r') as file:
            characters = json.load(file)
            if character in characters:
                return characters.index(character)
            else:
                raise Exception(f"{character} was not found in the model. Expected one of: {characters}")

    def execute_program(user_text, character, style_text, language, emotion, emotion_energy, audio_temperature,
              audio_top_k, audio_top_p, repetition_penalty, rvq_codebook_layers, gpu_id, tempdir):
        global MAX_NEW_TOKENS

        client = httpclient.InferenceServerClient(url="localhost:8000", network_timeout=300)

        # Wrap inputs
        user_text = np.array([user_text.encode("utf-8")], dtype='object')
        n_vq = np.array([rvq_codebook_layers], dtype=np.uint16)
        audio_temperature = np.array([audio_temperature], dtype=np.float32)
        audio_top_p = np.array([audio_top_p], dtype=np.float32)
        audio_top_k = np.array([audio_top_k], dtype=np.uint16)
        repetition_penalty = np.array([repetition_penalty], dtype=np.uint16)
        speaker_id = np.array([get_character_index(character)], dtype=np.uint16)
        language = np.array([language.encode("utf-8")], dtype='object')
        emotion_id = np.array([get_emotion_index(emotion, character)], dtype=np.uint16)
        emotion_energy = np.array([emotion_energy], dtype=np.float32)
        max_new_tokens = np.array([MAX_NEW_TOKENS], dtype=np.uint16)
        input_tensors = [
            httpclient.InferInput("text", shape=user_text.shape, datatype="BYTES").set_data_from_numpy(user_text),
            httpclient.InferInput("n_vq", n_vq.shape, datatype="UINT16").set_data_from_numpy(n_vq),
            httpclient.InferInput("audio_temperature", audio_temperature.shape, datatype="FP32").set_data_from_numpy(audio_temperature),
            httpclient.InferInput("audio_top_p", audio_top_p.shape, datatype="FP32").set_data_from_numpy(audio_top_p),
            httpclient.InferInput("audio_top_k", audio_top_k.shape, datatype="UINT16").set_data_from_numpy(audio_top_k),
            httpclient.InferInput("repetition_penalty", repetition_penalty.shape, datatype="UINT16").set_data_from_numpy(repetition_penalty),
            httpclient.InferInput("speaker_id", speaker_id.shape, datatype="UINT16").set_data_from_numpy(speaker_id),
            httpclient.InferInput("language", language.shape, datatype="BYTES").set_data_from_numpy(language),
            httpclient.InferInput("emotion_id", emotion_id.shape, datatype="UINT16").set_data_from_numpy(emotion_id),
            httpclient.InferInput("emotion_energy", emotion_energy.shape, datatype="FP32").set_data_from_numpy(emotion_energy),
            httpclient.InferInput("max_new_tokens", max_new_tokens.shape, datatype="UINT16").set_data_from_numpy(max_new_tokens),
        ]

        # Call Triton for inference
        query_response = client.infer(
            model_name=model_name,
            inputs=input_tensors
        )

        # Get output and write to file
        audio = query_response.as_numpy("audio")
        path = os.path.join(tempdir, 'output.wav')
        with wave.open(path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(audio.tobytes())

    def copy_output(tempdir, output_filename_sans_extension, session_id):
        array_output, sr_output = hsc.read_audio(hsc.get_single_file_with_extension(tempdir, '.wav'))
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py', description='A webservice interface for voice conversion with RVC')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(), help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    parser.add_argument('--max-new-tokens', type=int, default=250, help='Limits the length of audio that can be generated from a single prompt. 1 second = roughly 12.5 tokens')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = hsc.select_cache_implementation(args.cache_implementation)
    register_methods(cache, args.max_new_tokens)
    app.run(host='0.0.0.0', port=6582)
