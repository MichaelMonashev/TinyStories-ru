import argparse
import glob
import json
import os.path
import uuid
import sys

def _main(args):

    src_dir = args.src_dir
    dst_dir = args.dst_dir

    files = glob.glob(src_dir + "/*.jsonl")

    result_filename = None
    number_stories = 0

    for file in files:

        # подсчитываем количество историй в файле
        n = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                json.loads(line)
                n += 1
            print("Info:", file, "contain", n, "stories")

        if n==50000:
            print("number of stories == 50000, so skip", file)
            continue

        print("Proccess", file)

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)

                if number_stories >= 50000:
                    number_stories = 0
                    result_filename = None

                if result_filename is None:
                    result_filename = str(uuid.uuid4())
                    print("Create", result_filename)

                number_stories += 1

                filename = os.path.join(dst_dir,result_filename+'.jsonl')

                try:
                    with open(filename, 'a+', encoding='utf-8') as f:
                        json.dump(data, f, sort_keys=True, ensure_ascii=False)
                        f.write('\n')
                except:
                    print("Unexpected error", filename, sys.exc_info())
                    continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Join multiple JSONL files into one larger one.',
        epilog="Example: python3 join-jsonl.py --src-dir data/ --dst-dir data/"
    )
    parser.add_argument(
        '--src-dir',
        help='Path to directory with source jsonl files. Example: --src-dir data/',
        required=True)
    parser.add_argument(
        '--dst-dir',
        help='Path to directory with destination jsonl files. Example: --dst-dir data/',
        required=True)

    _main(parser.parse_args())
