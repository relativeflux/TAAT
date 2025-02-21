import argparse


def run():
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tape Archive Analysis Toolkit (TAAT)")
    parser.add_argument("config_file", help="Path to the config file feature.")
    args = parser.parse_args()

    run(args.config_file)