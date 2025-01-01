import argparse
import glob
import json

def _main(args):

    dir = args.dir

    files = glob.glob(dir + "/*.jsonl")

    ids = dict()

    for file in files:
        print(file)

        with open(file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    d = json.loads(line)

                    id = d['sha3_256']

                    if id in ids:
                        file2, line_number2 = ids[id]
                        print(f"Warning: dublicate find {file}:{line_number} - {file2}:{line_number2}")

                    ids[id] = (file, line_number)

                except UnicodeDecodeError:
                    # Обработка ошибок декодирования
                    print(f"Warning: Skipping line with decoding error {file} at line number {line_number}")
                except json.JSONDecodeError:
                    print(line)
                    print(f"Warning: JSON decoding error {file} at line number {line_number}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Join multiple JSONL files into one larger one.',
        epilog="Example: python3 join-jsonl.py --dir data/"
    )
    parser.add_argument(
        '--dir',
        help='Path to directory with jsonl files. Example: --dir data/',
        required=True)

    _main(parser.parse_args())
