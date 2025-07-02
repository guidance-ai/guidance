import argparse
import pathlib
import re


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--input_file", type=pathlib.Path, required=True, help="Path to input markdown file")
    parser.add_argument("--output_file", type=pathlib.Path, required=True, help="Path to output Python file")

    args = parser.parse_args()

    return args


def extract_python(markdown_text: str) -> str:
    START = "```python"
    END = "```"

    pattern = re.escape(START) + r"(.*?)" + re.escape(END)

    matches = re.findall(pattern, markdown_text, flags=re.DOTALL)

    for m in matches:
        print(m)

    return ""


def main():
    args = parse_arguments()

    assert args.input_file.exists(), f"{args.input_file} not found!"

    with open(args.input_file, "r") as readme_file:
        all_text = readme_file.read()

    extract_python(all_text)


if __name__ == "__main__":
    main()
