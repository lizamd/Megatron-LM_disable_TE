import os
import sqlite3
import textwrap
import subprocess

HEADER = f"{'>'*10} Legend Start {'<'*10}"
FOOTER = f"{'>'*10} Legend Ends  {'<'*10}"
KEYS_R = r"Keys: (?P<KEYS>[\w,]+)"


def make_header(keys):
    torch_ver = (
        subprocess.run(
            "pip3 show torch | grep Version | awk '{print $2}'",
            capture_output=True,
            shell=True,
        )
        .stdout.decode()
        .strip()
    )
    rocm_ver = (
        subprocess.run(
            "dpkg -l rocm-core | grep rocm-core | awk '{print $3}'",
            capture_output=True,
            shell=True,
        )
        .stdout.decode()
        .strip()
    )
    rocblas_ver = (
        subprocess.run(
            "dpkg -l rocblas | grep rocblas | awk '{print $3}'",
            capture_output=True,
            shell=True,
        )
        .stdout.decode()
        .strip()
    )
    hipblaslt_ver = (
        subprocess.run(
            "dpkg -l hipblaslt | grep hipblaslt | awk '{print $3}'",
            capture_output=True,
            shell=True,
        )
        .stdout.decode()
        .strip()
    )

    return (
        f"{HEADER}{os.linesep}"
        f"Torch Version: {torch_ver}{os.linesep}"
        f"rocm Version: {rocm_ver}{os.linesep}"
        f"rocBlas Version: {rocblas_ver}{os.linesep}"
        f"hipblasLT Version: {hipblaslt_ver}{os.linesep}"
        f"Keys: {keys}{os.linesep}"
        f"{FOOTER}{os.linesep}"
    )


def df_to_csv(df, file_name, *args, keys="", mode="w", **kwargs):
    """Prints df to csv with an aditioanl not of which fields are keys
    and not metrics. This is used for comparison purposes.
    """
    if os.path.dirname(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    file_exists = os.path.exists(file_name) and os.path.getsize(file_name) > 0
    not_appending = not ((mode == "a") and file_exists)

    if not_appending:
        with open(file_name, mode) as f:
            f.write(make_header(keys))
    df.to_csv(file_name, *args, header=not_appending, mode="a", **kwargs)


def query_rpd_database(db_path):
    from prettytable import PrettyTable

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Name, TotalCalls, TotalDuration, Ave, Percentage FROM top;"
        )
        rows = cursor.fetchall()

        if rows:
            table = PrettyTable()
            table.field_names = [
                "Name",
                "TotalCalls",
                "TotalDuration",
                "Ave",
                "Percentage",
            ]
            table.align = "l"

            for row in rows:
                wrapped_name = "\n".join(textwrap.wrap(row[0], 60))
                table.add_row([wrapped_name] + list(row[1:]))

            print(table)
        else:
            print("No data found in 'top' table.")

    except sqlite3.Error as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()


def is_float_try(f: str):
    try:
        return float(f)
    except ValueError:
        return f
