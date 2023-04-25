import argparse
import telnetlib
import time
import random
import datetime

VERSION = '1.0.0'

AM_START = 540000
AM_END = 1700000
AM_STEP = 10000

FM_START = 88100000
FM_END = 108000000
FM_STEP = 200000

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

	def get_strength(self):
		return self._write('l\n')

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
	args = parser.parse_args()

	if args.version:
		print(VERSION)
		return

	if not args.am and not args.fm:
		print('Error: AM/FM frequencies not enabled')
		return

	if args.speed <= 0:
		print('Error: Invalid scan speed')

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

	print('- {}ms interval'.format(args.speed))
	print('- {}dB squelch\n'.format(args.squelch))

	args.speed /= 1000
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

	start = datetime.datetime.now()
	print('Started @ {}'.format(start))

	freq = 0
	index = 0
	direction = 1

	try:
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
				time.sleep(args.speed / 2)
				# spooooky
				random.seed(gb.get_strength())
				time.sleep(args.speed / 2)
			else:
				time.sleep(args.speed)
	except KeyboardInterrupt:
		end = datetime.datetime.now()
		print('Ended @ {}\n'.format(end))
		print('Runtime {}'.format(end - start))

if __name__ == '__main__':
	main()