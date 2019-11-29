from tqdm.auto import tqdm

def _split_by_comma(string): return string.split(',')

_csv_header_to_column_names = _split_by_comma
_csv_row_to_values          = _split_by_comma

def split_by_columns(source_file_path, target_dir_path):
    target_files = []

    try:
        with open(source_file_path) as source_file:
            csv_header = source_file.readline().strip()
            column_names = _csv_header_to_column_names(csv_header)

            for column_name in column_names:
                file = open(f'{target_dir_path}/{column_name}.csv', 'w')
                file.write(column_name + '\n')
                target_files.append(file)

            for row in tqdm(source_file):
                values = _csv_row_to_values(row.strip())
                assert len(values) == len(column_names)

                for value, file in zip(values, target_files):
                    if value == '': value = 'NULL'

                    file.write(value + '\n')
    finally:
        for file in target_files:
            file.close()
