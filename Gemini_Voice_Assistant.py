# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyaudio",
#     "websockets",
# ]
# ///
import asyncio
import base64
import json
import os
import pyaudio
from websockets.asyncio.client import connect

inputAudio = None
isPlaying = False

class GeminiVoiceAssistant:
    def __init__(self):
        self._audio_queue = asyncio.Queue()
        self._api_key = os.environ.get("GEMINI_API_KEY")
        self._model = "gemini-2.0-flash-exp"
        self._uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self._api_key}"
        # Audio settings
        self._FORMAT = pyaudio.paInt16
        self._CHANNELS = 1
        self._CHUNK = 512
        self._RATE = 16000

    async def _connect_to_gemini(self):
        return await connect(
            self._uri, additional_headers={"Content-Type": "application/json"}
        )

    async def _start_audio_streaming(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._capture_audio())
            tg.create_task(self._stream_audio())
            tg.create_task(self._play_response())

    async def _capture_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self._FORMAT,
            channels=self._CHANNELS,
            rate=self._RATE,
            input=True,
            frames_per_buffer=self._CHUNK,
        )

        while True:
            global isPlaying
            data = await asyncio.to_thread(stream.read, self._CHUNK)
            if isPlaying:
                continue

            await self._ws.send(
                json.dumps(
                    {
                        "realtime_input": {
                            "media_chunks": [
                                {
                                    "data": base64.b64encode(data).decode(),
                                    "mime_type": "audio/pcm",
                                }
                            ]
                        }
                    }
                )
            )

    async def _stream_audio(self):
        async for msg in self._ws:
            response = json.loads(msg)
            try:
                audio_data = response["serverContent"]["modelTurn"]["parts"][0][
                    "inlineData"
                ]["data"]
                self._audio_queue.put_nowait(base64.b64decode(audio_data))
            except KeyError:
                pass
            try:
                turn_complete = response["serverContent"]["turnComplete"]
            except KeyError:
                pass
            else:
                global isPlaying
                if turn_complete:
                    isPlaying = False
                    # If you interrupt the model, it sends an end_of_turn. For interruptions to work, we need to empty out the audio queue
                    print("\nEnd of turn, back to listening")
                    while not self._audio_queue.empty():
                        self._audio_queue.get_nowait()

    async def _play_response(self):
        # global inputAudio
        # # mute mic while playing response and few milliseconds after
        # device_index = inputAudio.get_default_input_device_info()['index']
        # # Mute the mic by setting the input volume to 0
        # inputAudio.set_input_device_volume(device_index, 0.0)
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self._FORMAT, channels=self._CHANNELS, rate=24000, output=True
        )
        while True:
            global isPlaying
            data = await self._audio_queue.get()
            isPlaying = True
            print("Playing response")
            await asyncio.to_thread(stream.write, data)

    async def start(self):
        self._ws = await self._connect_to_gemini()
        await self._ws.send(json.dumps({"setup": {"model": f"models/{self._model}"}}))
        await self._ws.recv(decode=False)
        print("Connected to Gemini, You can start talking now")
        await self._start_audio_streaming()


if __name__ == "__main__":
    client = GeminiVoiceAssistant()
    asyncio.run(client.start())
