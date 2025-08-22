import pandas as pd
import os
from tkinter import filedialog

#---------------------------------------- Open Dataset ---------------------------------------#                 
def open_csv(rows: int = 50, preview_rows: int = 100) -> pd.DataFrame:
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        raise SystemExit("\n---> open_csv Error 1: CSV file not loaded, try again!")
    else:
        file_name = os.path.basename(file_path)
        print(f"\nCSV file: {file_name} loaded successfully!")

    print(f"File path: {file_path}")
    directory = os.path.dirname(file_path)
    print(f"Directory: {directory}")

    print(f"\n---> Processing first {preview_rows} rows using chunksize...\n")

    # Leer las primeras 'preview_rows' usando chunksize
    chunk_iter = pd.read_csv(file_path, chunksize=preview_rows)
    first_chunk = next(chunk_iter)

    # Seleccionar solo los primeros 'rows' del primer chunk
    df = first_chunk.iloc[:rows, :]

    # Generar nombre de salida
    base_name = os.path.splitext(file_name)[0]
    output_filename = f"{base_name}_{rows}_rows.csv"
    export_path = os.path.join(directory, output_filename)

    # Guardar CSV reducido
    df.to_csv(export_path, index=False)
    print(f"\nProcessed CSV saved at: {export_path}")

    return df

if __name__ == '__main__':
    open_csv()
