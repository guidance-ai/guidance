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
    # This function contains 95% CoPilot vibes by volume
    START = "```python"
    END = "```"

    pattern = re.escape(START) + r"(.*?)" + re.escape(END)

    matches = re.findall(pattern, markdown_text, flags=re.DOTALL)

    result = ""
    for m in matches:
        result += str(m)

    return result


def main():
    args = parse_arguments()

    assert args.input_file.exists(), f"{args.input_file} not found!"

    with open(args.input_file, "r") as readme_file:
        all_text = readme_file.read()
    print("Read input")

    output_python = extract_python(all_text)
    print("Extracted Python code")

    with open(args.output_file, "w") as py_file:
        py_file.write(output_python)
    print("Output file written")


if __name__ == "__main__":
    main()
