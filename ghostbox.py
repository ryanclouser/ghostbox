import argparse
import telnetlib
import time
import random
import datetime
import glob
import os
import csv
from pathlib import Path

VERSION = '1.0.1'

AM_START = 540000
AM_END = 1700000
AM_STEP = 10000

FM_START = 88100000
FM_END = 108000000
FM_STEP = 200000

WAV_PATH = os.path.join(Path.home(), 'gqrx_*.wav')


try:
	import torch
	import torchaudio
	import torch.multiprocessing as mp
	import sounddevice as sd

	bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = bundle.get_model().to(device)
	resampler = torchaudio.transforms.Resample(48_000, 16_000)

	tts_model = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='en', speaker='v3_en', verbose=False)[0]
	tts_model.to(device)
	TORCH = True
except Exception as e:
	TORCH = False
	print(e)


# https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html
class GreedyCTCDecoder(torch.nn.Module):
	def __init__(self, labels, blank=0):
		super().__init__()
		self.labels = labels
		self.blank = blank

	def forward(self, emission: torch.Tensor) -> list[str]:
		"""Given a sequence emission over labels, get the best path
		Args:
		emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

		Returns:
		List[str]: The resulting transcript
		"""
		indices = torch.argmax(emission, dim=-1)  # [num_seq,]
		indices = torch.unique_consecutive(indices, dim=-1)
		indices = [i for i in indices if i != self.blank]
		joined = ''.join([self.labels[i] for i in indices])
		return joined.replace('|', ' ').strip().lower().split()


def infer(path):
	waveform, sample_rate = torchaudio.load(path)
	waveform = resampler(waveform).squeeze()

	with torch.inference_mode():
		features, _ = model.extract_features(waveform)
	with torch.inference_mode():
		emission, _ = model(waveform)

	decoder = GreedyCTCDecoder(labels=bundle.get_labels())
	return decoder(emission[0])


def tts(text, blocking=True):
	sample_rate = 48000
	speaker = 'en_0'

	audio = tts_model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
	sd.play(audio, sample_rate, blocking=blocking)


def process(q, use_wordlist, use_tts):
	words = []
	if use_wordlist:
		try:
			with open('word-freq-top5000.csv', newline='') as csvfile:
				reader = csv.reader(csvfile, delimiter=',', quotechar='"')
				for row in reader:
					words.append(row[1])
		except Exception as e:
			print(e)

	while True:
		f = q.get()
		result = infer(f)
		data = []

		for word in result:
			if not words or word in words:
				data.append(word)

		if data:
			print('{} -> {}'.format(datetime.datetime.now(), ' '.join(data)))
			if use_tts:
				tts('.'.join(data) + '.')

		os.remove(f)


class Ghostbox:
	def __init__(self, ip, port):
		self.tn = telnetlib.Telnet(ip, port)

	def _write(self, msg):
		self.tn.write(msg.encode('ascii'))
		return self.tn.read_some().decode('ascii').strip()

	def set_squelch(self, sql):
		return self._write('L SQL {}\n'.format(sql))

	def set_freq(self, freq):
		return self._write('F {}\n'.format(freq))

	def set_mode(self, mode):
		return self._write('M {}\n'.format(mode))

	def set_record(self, mode):
		return self._write('U RECORD {}\n'.format(mode))

	def get_strength(self):
		return float(self._write('l\n'))


def gb_freq(gb, freq, mode=None):
	if mode:
		gb.set_mode(mode)
	gb.set_freq(freq)


def main():
	parser = argparse.ArgumentParser(description='Python Ghostbox')
	parser.add_argument('--version', action='store_true', help='Version')
	parser.add_argument('--ip', type=str, default='127.0.0.1', help='GQRX IP')
	parser.add_argument('--port', type=int, default=7356, help='GQRX port')
	parser.add_argument('--fm', action='store_true', help='Enable FM radio scanning')
	parser.add_argument('--am', action='store_true', help='Enable AM radio scanning')
	parser.add_argument('--speed', type=int, default=150, help='Scanning speed in milliseconds')
	parser.add_argument('--squelch', type=int, default=-30, help='Squelch')
	parser.add_argument('--random', action='store_true', help='Random scanning')
	parser.add_argument('--forward', action='store_true', help='Forward scanning')
	parser.add_argument('--backward', action='store_true', help='Backward scanning')
	parser.add_argument('--bounce', action='store_true', default=True, help='Bounce scanning')
	parser.add_argument('--record', action='store_true', help='Record and process audio')
	parser.add_argument('--record-interval', type=float, default=5.0, help='Time for each audio recording')
	parser.add_argument('--wordlist', action='store_true', help='Use a wordlist after audio is processed')
	parser.add_argument('--tts', action='store_true', help='Use text to speech')
	args = parser.parse_args()

	if args.version:
		print(VERSION)
		return

	if not args.am and not args.fm:
		print('Error: AM/FM frequencies not enabled')
		return

	if args.speed < 0:
		print('Error: Invalid scan speed')
	elif args.speed > 0:
		args.speed /= 1000

	print("   _____ _               _   _               ")
	print("  / ____| |             | | | |              ")
	print(" | |  __| |__   ___  ___| |_| |__   _____  __")
	print(" | | |_ | '_ \\ / _ \\/ __| __| '_ \\ / _ \\ \\/ /")
	print(" | |__| | | | | (_) \\__ \\ |_| |_) | (_) >  < ")
	print("  \\_____|_| |_|\\___/|___/\\__|_.__/ \\___/_/\\_\\\n")
	print('v{}\n'.format(VERSION))

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

	if args.speed > 0:
		print('- {}s interval'.format(args.speed))
	else:
		print('- Random interval')

	print('- {}dB squelch'.format(args.squelch))

	if args.record:
		if TORCH:
			print('- Recording')
			print('- {}s record interval'.format(args.record_interval))
		else:
			args.record = False
			print('Torch is not installed, cannot process recordings')

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

	if args.record:
		ctx = mp.get_context('spawn')
		q = ctx.Queue()
		p = ctx.Process(target=process, args=(q, args.wordlist, args.tts))
		p.start()

	start = datetime.datetime.now()
	print('Started @ {}\n'.format(start))

	try:
		freq = 0
		index = 0
		direction = 1

		if args.record:
			gb.set_record(0)
			for f in glob.glob(WAV_PATH):
				print('Removing {}'.format(f))
				os.remove(f)
			print()

			record_start = time.time()
			gb.set_record(1)

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

			if args.record and time.time() - record_start >= args.record_interval:
				gb.set_record(0)
				files = glob.glob(WAV_PATH)
				files.sort(key=os.path.getmtime)
				q.put(files[-1])
				record_start = time.time()
				gb.set_record(1)

			if freq < FM_START:
				gb_freq(gb, freq, 'AM' if mode else None)
			else:
				gb_freq(gb, freq, 'WFM' if mode else None)

			if args.random:
				time.sleep(args.speed / 2)
				# spooooky
				random.seed(gb.get_strength())
				time.sleep(args.speed / 2)
			elif args.speed > 0:
				time.sleep(args.speed)
			else:
				time.sleep(0.5 + gb.get_strength() / 100)
	except KeyboardInterrupt:
		pass

	end = datetime.datetime.now()
	print('Ended @ {}\n'.format(end))
	print('Runtime {}'.format(end - start))

	if args.record:
		p.join()


if __name__ == '__main__':
	main()