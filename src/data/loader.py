import pandas as pd
import pathlib
import io
from typing import Union, Optional

def load_data(file_source: Union[str, pathlib.Path, io.BytesIO, io.StringIO], 
              file_name: Optional[str] = None, 
              **kwargs) -> pd.DataFrame:
    """
    Loads a structured dataset from CSV or Excel formats.
    
    Parameters:
    -----------
    file_source : Union[str, pathlib.Path, io.BytesIO, io.StringIO]
        The path to the file or a file-like object (e.g. Streamlit UploadedFile).
    file_name : Optional[str]
        The file name (required to determine format if file_source is a file-like object).
    **kwargs : Dict[str, Any]
        Additional arguments passed to pd.read_csv or pd.read_excel.

    Returns:
    --------
    pd.DataFrame
        Loaded Pandas DataFrame.

    Raises:
    -------
    ValueError
        If format is unsupported, the file is empty, or the file is malformed/unreadable.
    FileNotFoundError
        If the file path does not exist.
    """
    # 1. Determine format from source or file_name
    suffix = ""
    if isinstance(file_source, (str, pathlib.Path)):
        path = pathlib.Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"The specified file path does not exist: {file_source}")
        suffix = path.suffix.lower()
    elif file_name is not None:
        suffix = pathlib.Path(file_name).suffix.lower()
    elif hasattr(file_source, 'name'):
        suffix = pathlib.Path(file_source.name).suffix.lower()
        
    if not suffix:
        raise ValueError("Could not determine file format. Please provide a file with a valid extension (.csv, .xlsx).")

    if suffix not in ['.csv', '.xlsx']:
        raise ValueError(f"Unsupported file format: {suffix}. Only CSV (.csv) and Excel (.xlsx) files are supported.")

    # 2. Check for empty files
    if isinstance(file_source, (str, pathlib.Path)):
        if pathlib.Path(file_source).stat().st_size == 0:
            raise ValueError("The provided file is empty (0 bytes).")
    elif isinstance(file_source, io.BytesIO):
        file_source.seek(0, io.SEEK_END)
        size = file_source.tell()
        file_source.seek(0)
        if size == 0:
            raise ValueError("The uploaded file is empty.")
    elif isinstance(file_source, io.StringIO):
        val = file_source.getvalue()
        if not val.strip():
            raise ValueError("The uploaded file is empty.")

    # 3. Read data
    try:
        if suffix == '.csv':
            # Handle encoding issues gracefully
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            last_err = None
            for encoding in encodings:
                try:
                    if hasattr(file_source, 'seek'):
                        file_source.seek(0)
                    
                    kwargs_copy = kwargs.copy()
                    if 'encoding' not in kwargs_copy:
                        kwargs_copy['encoding'] = encoding
                    
                    df = pd.read_csv(file_source, **kwargs_copy)
                    
                    if df.empty:
                        raise ValueError("The file contains no columns or data.")
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    last_err = e
                    continue
            raise ValueError(f"Failed to parse CSV file. The file may be malformed or use an unsupported encoding. Original error: {str(last_err)}")
            
        elif suffix == '.xlsx':
            if hasattr(file_source, 'seek'):
                file_source.seek(0)
            df = pd.read_excel(file_source, **kwargs)
            if df.empty:
                raise ValueError("The file contains no columns or data.")
            return df
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to load data from file: {str(e)}")
