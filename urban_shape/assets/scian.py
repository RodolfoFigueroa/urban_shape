import string
import unicodedata

import dagster as dg
import pandas as pd

from nltk.corpus import stopwords
from pathlib import Path
from urban_shape.resources import PathResource


def remove_punctuation(text) -> str:
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_accents(text: str) -> str:
    return ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('spanish'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def process_text(text: str) -> str:
    text = text.strip().rstrip("T").casefold()
    text = remove_punctuation(text)
    text = remove_accents(text)
    text = remove_stopwords(text)
    return text


@dg.asset(
    name="scian",
    io_manager_key="dataframe_manager",
    group_name="scian"
)
def scian(path_resource: PathResource) -> pd.DataFrame:
    scian_path = Path(path_resource.scian_path)

    return (
        pd.read_excel(
            scian_path / "scian_2023_categorias_y_productos.xlsx", 
            sheet_name="SUBSECTOR", 
            skiprows=1, 
            usecols=["Código", "Título"]
        )
        .dropna(subset="Código")
        .assign(
            Código=lambda df: df["Código"].astype(int),
        )
        .rename(columns={"Código": "subcategory", "Título": "text"})
        .assign(text=lambda df: df["text"].apply(process_text))
    )