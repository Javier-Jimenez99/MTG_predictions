import typer
import os

def split_csv(input_file: str, output_folder: str, file_lines:int=52000):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'r') as f:
        header = f.readline()
        out=None
        for i, line in enumerate(f):
            if i % file_lines == 0:
                if i > 0:
                    out.close()
                out = open(f"{output_folder}/part_{i//file_lines}.csv", 'w')
                out.write(header)
            out.write(line)

        out.close()

if __name__ == "__main__":
    typer.run(split_csv)