import pandas as pd
import numpy as np
import argparse
import re
from afo.tools.utils import HEADER, FOOTER, KEYS_R


def search_file(file, header_p, footer_p, keys_p):
    header_line = -1
    footer_line = -1
    keys = None
    with open(file) as f:
        line_counter = 0
        while line := f.readline():
            line = line.rstrip()
            if header_p.match(line):
                header_line = line_counter
            elif footer_p.match(line):
                footer_line = line_counter
                break
            elif match := keys_p.match(line):
                keys = match.group("KEYS")

            line_counter += 1
    return (header_line, footer_line, keys)


class csvMerger:
    """Merges and compares two csv formatted by df_to_csv func"""

    def __init__(
        self,
        first_csv: str,
        second_csv: str,
        output_csv: str = "merge.xlsx",
        first_suffix: str = "_x",
        second_suffix: str = "_y",
    ):
        self.first = first_csv
        self.second = second_csv
        self.output = output_csv
        self.first_suffix = first_suffix
        self.second_suffix = second_suffix

    def df_to_excel(self, df):
        writer = pd.ExcelWriter(self.output, engine="xlsxwriter")
        sheet_name = "Sheet1"
        df.to_excel(writer, sheet_name="Sheet1")
        worksheet = writer.sheets[sheet_name]

        # Apply a conditional format to the cell range.
        max_row, max_col = df.shape
        df.to_excel(writer, sheet_name=sheet_name)

        idx = 1
        for column in df.columns:
            if column.endswith("%_change"):
                worksheet.conditional_format(
                    0, idx, max_row, idx, {"type": "3_color_scale"}
                )
            idx += 1

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()

    def geomean(self, x):
        try:
            _x = x.replace(0, np.nan)
            return np.exp(np.mean(np.log(_x)))
        except:  # noqa: E722
            return 0

    def calculate_geomeans(self, df):
        df.loc[len(df)] = df.apply(lambda x: self.geomean(x), axis=0)

    def get_headers(self):
        header = re.compile(HEADER)
        footer = re.compile(FOOTER)
        keys = re.compile(KEYS_R)

        self.f_header, self.f_footer, self.f_keys = search_file(
            self.first, header, footer, keys
        )
        self.s_header, self.s_footer, self.s_keys = search_file(
            self.second, header, footer, keys
        )

        assert self.f_header != self.f_footer
        assert self.s_header != self.s_footer
        assert self.f_keys == self.s_keys

    def calculate_change(self, df):
        columns_to_compare = [
            c[0 : -len(self.first_suffix)]
            for c in df.columns
            if c.endswith(self.first_suffix)
        ]
        for column in columns_to_compare:
            f_column = df[f"{column}{self.first_suffix}"]
            s_column = df[f"{column}{self.second_suffix}"]
            perc_diffs = (s_column - f_column) / f_column

            # TODO: can we get the loc from the obj itself?
            insert_idx = df.columns.get_loc(f"{column}{self.second_suffix}") + 1

            df.insert(insert_idx, f"{column}_%_change", perc_diffs)

    def merge_csv(self):
        self.get_headers()

        f_df = pd.read_csv(self.first, skiprows=self.f_footer + 1)
        s_df = pd.read_csv(self.second, skiprows=self.s_footer + 1)

        keys = self.f_keys.split(",")
        merged_df = pd.merge(
            f_df,
            s_df,
            how="outer",
            on=keys,
            suffixes=[self.first_suffix, self.second_suffix],
        )

        self.calculate_geomeans(merged_df)
        merged_df.rename(
            {merged_df.shape[0] - 1: "Column Geomeans"}, inplace=True, errors="raise"
        )

        self.calculate_change(merged_df)
        columns = merged_df.columns
        ordered_columns = keys
        ordered_columns.extend(
            sorted([c for c in columns if str(c) not in keys], reverse=True)
        )
        merged_df = merged_df.reindex(columns=ordered_columns)

        return merged_df


def main():
    parser = argparse.ArgumentParser(description="Merge and compare CSVs")

    parser.add_argument(
        "--output",
        metavar="OUTPUT_FILE",
        default="merged.csv",
        help="Name of the output file to write (Defaults to merged.csv)",
    )

    parser.add_argument(
        "first_csv",
        metavar="FIRST_FILE",
        # type=argparse.FileType("r"),
        help="path to the first CSV",
    )

    parser.add_argument(
        "second_csv",
        metavar="SECOND_FILE",
        # type=argparse.FileType("r"),
        help="path to the second csv",
    )

    parser.add_argument(
        "--first_suffix",
        default="_x",
        help="Distinguishing suffix for the first csv",
    )

    parser.add_argument(
        "--second_suffix",
        default="_y",
        help="Distinguishing suffix for the second csv",
    )
    args = parser.parse_args()
    merger_class = csvMerger(
        args.first_csv,
        args.second_csv,
        args.output,
        args.first_suffix,
        args.second_suffix,
    )

    merged_df = merger_class.merge_csv()
    merger_class.df_to_excel(merged_df)


if __name__ == "__main__":
    main()
