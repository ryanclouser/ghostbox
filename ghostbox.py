#!/usr/bin/env python3
import argparse
import time
import random
import datetime
import glob
import os
import csv
from pathlib import Path
import socket
from TimeseriesCache import TimeseriesCache

__author__ = 'Ryan Clouser'
__copyright__ = 'Copyright (c) Ryan Clouser 2023'
__license__ = 'MIT'
__version__ = '1.1.1'


AM_START = 540000
AM_END = 1700000
AM_STEP = 10000

FM_START = 88100000
FM_END = 108000000
FM_STEP = 200000


try:
	import torch
	import torchaudio
	from torchaudio.io import StreamReader
	import torch.multiprocessing as mp
	import sounddevice as sd

	backend = 'cpu'
	if torch.cuda.is_available():
		backend = 'cuda'
	elif torch.backends.mps.is_available():
		backend = 'mps'

	device = torch.device(backend)
	tts_model = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='en', speaker='v3_en', verbose=False)[0]
	tts_model.to(device)

	TORCH = True
except ImportError as e:
	TORCH = False
	print(e)


class Pipeline:
	"""Build inference pipeline from RNNTBundle.

	Args:
		bundle (torchaudio.pipelines.RNNTBundle): Bundle object
		beam_width (int): Beam size of beam search decoder.
	"""

	def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, beam_width: int = 10):
		self.bundle = bundle
		self.feature_extractor = bundle.get_streaming_feature_extractor()
		self.decoder = bundle.get_decoder()
		self.token_processor = bundle.get_token_processor()

		self.beam_width = beam_width

		self.state = None
		self.hypothesis = None

	def infer(self, segment: torch.Tensor) -> str:
		"""Perform streaming inference"""
		features, length = self.feature_extractor(segment)
		hypos, self.state = self.decoder.infer(
			features, length, self.beam_width, state=self.state, hypothesis=self.hypothesis
		)
		self.hypothesis = hypos[0]
		return self.token_processor(self.hypothesis[0]).strip()


class ContextCacher:
	"""Cache the end of input data and prepend the next input data with it.

	Args:
		segment_length (int): The size of main segment.
		If the incoming segment is shorter, then the segment is padded.
		context_length (int): The size of the context, cached and appended.
	"""

	def __init__(self, segment_length: int, context_length: int):
		self.segment_length = segment_length
		self.context_length = context_length
		self.context = torch.zeros([context_length])

	def __call__(self, chunk: torch.Tensor):
		if chunk.size(0) < self.segment_length:
			chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
		chunk_with_context = torch.cat((self.context, chunk))
		self.context = chunk[-self.context_length :]
		return chunk_with_context


class UDPWrapper:
	def __init__(self, obj):
		self.obj = obj

	def read(self, n):
		return self.obj.recvfrom(n)[0]


def stream(q, ip, segment_length, sample_rate):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	while True:
		try:
			s.bind((ip, 7355))
			break
		except Exception as e:
			print(e)
			time.sleep(2)

	streamer = StreamReader(UDPWrapper(s), format='s16le')
	streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

	stream_iterator = streamer.stream(timeout=-1, backoff=1.0)

	try:
		while True:
			(chunk,) = next(stream_iterator)
			q.put(chunk)
	except KeyboardInterrupt:
		pass


def tts(q, reverb):
	sample_rate = 48000
	speaker = 'en_0'

	# https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html
	effects = [
		["speed", "0.90"],
		["rate", f"{sample_rate}"]
	]

	if reverb:
		effects.append(["reverb", "-w"])

	try:
		while True:
			text = q.get()
			audio = tts_model.apply_tts(text=','.join(text) + '.', speaker=speaker, sample_rate=sample_rate)
			audio = torchaudio.sox_effects.apply_effects_tensor(audio.unsqueeze(0), sample_rate, effects)[0]
			sd.play(audio[0], sample_rate, blocking=True)
	except KeyboardInterrupt:
		pass


def playback(q, reverb, speed, sample_rate):
	effects = [
		["speed", f"{speed}"],
		["rate", f"{sample_rate}"]
	]

	if reverb:
		effects.append(["reverb", "-w"])

	try:
		while True:
			waveform = q.get()
			audio = torchaudio.sox_effects.apply_effects_tensor(waveform.unsqueeze(0), sample_rate, effects)[0]
			sd.play(audio[0], sample_rate, blocking=True)
	except KeyboardInterrupt:
		pass


def process(ip, interval, use_wordlist, use_tts, long_words, reverb, use_playback, playback_speed):
	bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

	sample_rate = bundle.sample_rate
	segment_length = bundle.segment_length * bundle.hop_length
	context_length = bundle.right_context_length * bundle.hop_length
	pipeline = Pipeline(bundle)

	cacher = ContextCacher(segment_length, context_length)

	words = []
	if use_wordlist:
		try:
			with open('word-freq-top5000.csv', newline='') as csvfile:
				reader = csv.reader(csvfile, delimiter=',', quotechar='"')
				for row in reader:
					words.append(row[1])
		except Exception as e:
			print(e)

	ctx = mp.get_context("spawn")
	q1 = ctx.Queue()
	p1 = ctx.Process(target=stream, args=(q1, ip, segment_length, sample_rate))
	p1.start()

	if use_tts:
		q2 = ctx.Queue()
		p2 = ctx.Process(target=tts, args=(q2, reverb))
		p2.start()
	elif use_playback:
		q2 = ctx.Queue()
		p2 = ctx.Process(target=playback, args=(q2, reverb, playback_speed, sample_rate))
		p2.start()

	@torch.inference_mode()
	def infer():
		c = TimeseriesCache(ttl=10.0)
		start = 0.0
		t = 0.0
		sentence = []

		try:
			while True:
				chunk = q1.get()
				segment = cacher(chunk[:, 0])
				c.add(time.time(), segment)

				transcript = pipeline.infer(segment)
				if transcript:
					if not words or transcript in words:
						t = time.time()
						if not start:
							start = t
						sentence.append(transcript)

				if sentence and time.time() - t > 2.5:
					if long_words:
						sentence = [word for word in sentence if len(word) > 2]
					if sentence:
						if use_tts:
							q2.put(sentence)
						elif use_playback:
							q2.put(torch.cat(c[start - interval:t + interval], dim=0))
						print('{} -> {}'.format(datetime.datetime.now(), ' '.join(sentence)))
						sentence = []
						start = 0.0
		except KeyboardInterrupt:
			pass

	infer()

	p1.join()
	if use_tts or use_playback:
		p2.join()


class Ghostbox:
	def __init__(self, ip, port):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		while True:
			try:
				self.s.connect((ip, port))
				break
			except Exception as e:
				print(e)
				time.sleep(2.0)

	def __write(self, msg):
		self.s.send(msg.encode('ascii'))
		return self.s.recv(1024).decode('ascii').strip()

	def set_squelch(self, sql):
		return self.__write('L SQL {}\n'.format(sql))

	def set_freq(self, freq):
		return self.__write('F {}\n'.format(freq))

	def set_mode(self, mode):
		return self.__write('M {}\n'.format(mode))

	def set_record(self, mode):
		return self.__write('U RECORD {}\n'.format(mode))

	def get_strength(self):
		return float(self.__write('l\n'))


def gb_freq(gb, freq, mode=None):
	if mode:
		gb.set_mode(mode)
	gb.set_freq(freq)


def main():
	parser = argparse.ArgumentParser(description='Ghostbox')
	parser.add_argument('-v', '--version', action='store_true', help='display the program version')
	parser.add_argument('--ip', type=str, default='127.0.0.1', help='gqrx IP')
	parser.add_argument('-p', '--port', type=int, default=7356, help='gqrx port')
	parser.add_argument('--fm', action='store_true', help='enable FM radio scanning')
	parser.add_argument('--am', action='store_true', help='enable AM radio scanning')
	parser.add_argument('-i', '--interval', type=float, default=0.15, help='scanning interval in seconds')
	parser.add_argument('-s', '--squelch', type=int, default=-45, help='squelch')
	parser.add_argument('--random', action='store_true', help='random scanning')
	parser.add_argument('--forward', action='store_true', help='forward scanning')
	parser.add_argument('--backward', action='store_true', help='backward scanning')
	parser.add_argument('--bounce', action='store_true', default=True, help='bounce scanning')
	parser.add_argument('--stt', action='store_true', help='enable speech to text')
	parser.add_argument('-w', '--wordlist', action='store_true', help='use a wordlist after audio is processed')
	parser.add_argument('--tts', action='store_true', help='enable text to speech')
	parser.add_argument('-r', '--reverb', action='store_true', help='apply reverb effect to TTS')
	parser.add_argument('-l', '--long-words', action='store_true', help='hide short words from the output')
	parser.add_argument('-pb', '--playback', action='store_true', help='playback transcript audio')
	parser.add_argument('-ps', '--playback-speed', type=float, default=1.0, help='playback speed')
	args = parser.parse_args()

	if args.version:
		print(__version__)
		return

	if not args.am and not args.fm:
		print('Error: AM/FM frequencies not enabled')
		return

	if args.interval < 0:
		print('Error: Invalid scan interval')
		return

	if args.tts and args.playback:
		print('Error: TTS and playback cannot both be enabled')
		return

	print("   _____ _               _   _               ")
	print("  / ____| |             | | | |              ")
	print(" | |  __| |__   ___  ___| |_| |__   _____  __")
	print(" | | |_ | '_ \\ / _ \\/ __| __| '_ \\ / _ \\ \\/ /")
	print(" | |__| | | | | (_) \\__ \\ |_| |_) | (_) >  < ")
	print("  \\_____|_| |_|\\___/|___/\\__|_.__/ \\___/_/\\_\\\n")
	print('v{}\n'.format(__version__))

	if args.am and args.fm:
		print('- AM/FM enabled')
	elif args.am:
		print('- AM enabled')
	elif args.fm:
		print('- FM enabled')

	if args.random:
		print('- Random')
	elif args.forward:
		print('- Forward')
	elif args.backward:
		print('- Backward')
	else:
		print('- Bounce')

	if args.interval > 0:
		print('- {}s interval'.format(args.interval))
	else:
		print('- Random interval')

	print('- {}dB squelch'.format(args.squelch))

	if TORCH:
		if args.stt:
			print('- Speech to Text')
			if args.tts:
				print('- Text to Speech')
			if args.long_words:
				print('- Hide short words')

	print()

	frequencies = []

	if args.am:
		for freq in range(AM_START, AM_END, AM_STEP):
			frequencies.append(freq)

	if args.fm:
		for freq in range(FM_START, FM_END, FM_STEP):
			frequencies.append(freq)

	gb = Ghostbox(args.ip, args.port)
	gb.set_squelch(args.squelch)

	mode = (args.am == args.fm)
	if not mode:
		if args.am:
			gb.set_mode('AM')
		else:
			gb.set_mode('WFM')

	if args.stt:
		ctx = mp.get_context('spawn')
		p = ctx.Process(target=process, args=(args.ip, args.interval, args.wordlist, args.tts, args.long_words, args.reverb, args.playback, args.playback_speed))
		p.start()

	start = datetime.datetime.now()
	print('Started @ {}\n'.format(start))

	try:
		freq = 0
		index = 0
		direction = 1

		while True:
			if args.random:
				freq = random.choice(frequencies)
			elif args.forward:
				if index >= len(frequencies):
					index = 0
				freq = frequencies[index]
				index += 1
			elif args.backward:
				index -= 1
				if index < 0:
					index = len(frequencies) - 1
				freq = frequencies[index]
			elif args.bounce:
				if (index <= 0 and direction < 0) or index >= len(frequencies) - 1:
					direction *= -1
				freq = frequencies[index]
				index += direction

			if freq < FM_START:
				gb_freq(gb, freq, 'AM' if mode else None)
			else:
				gb_freq(gb, freq, 'WFM' if mode else None)

			if args.random:
				time.sleep(args.interval / 2)
				# spooooky
				random.seed(gb.get_strength())
				time.sleep(args.interval / 2)
			elif args.interval > 0:
				time.sleep(args.interval)
			else:
				time.sleep(0.5 + gb.get_strength() / 100)
	except KeyboardInterrupt:
		pass

	end = datetime.datetime.now()
	print('Ended @ {}\n'.format(end))
	print('Runtime {}'.format(end - start))

	if args.stt:
		p.join()


if __name__ == '__main__':
	main()