import dagster as dg
import geopandas as gpd
import pandas as pd

from upath import UPath as Path
from urban_shape.resources import PathResource
from cfc_core_utils import gdal_azure_session, storage_options
import fsspec
import shutil
import tempfile
from pathlib import Path as LocalPath

class BaseManager(dg.ConfigurableIOManager):
    extension: str
    path_resource: dg.ResourceDependency[PathResource]

    def _get_path(self, context: dg.InputContext | dg.OutputContext) -> Path:
        out_path = Path(self.path_resource.data_path) / "generated"
        fpath = out_path / "/".join(context.asset_key.path)

        if context.has_asset_partitions:
            final_path = fpath / context.asset_partition_key
            final_path = final_path.with_suffix(final_path.suffix + self.extension)
        else:
            final_path = fpath.with_suffix(fpath.suffix + self.extension)

        return final_path
    
    def _makedirs(self, p: Path) -> None:
        # Only create dirs for local FS; object stores don't need them
        if getattr(p, "protocol", "file") == "file":
            p.parent.mkdir(parents=True, exist_ok=True) 
            
    def _as_vsi(self, p: Path) -> str:
        # For GDAL/Fiona/Rasterio on Azure
        return str(p).replace("az://", "/vsiaz/") 

class DataFrameManager(BaseManager):
    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        path = self._get_path(context)
        self._makedirs(path)
        obj.to_csv(path, index=False, storage_options=storage_options(path))

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        path = self._get_path(context)
        return pd.read_csv(path, storage_options=storage_options(path))


class GeoDataFrameManager(BaseManager):
    def handle_output(self, context: dg.OutputContext, obj: gpd.GeoDataFrame) -> None:
        path = self._get_path(context)
        self._makedirs(path)

        # Stage to local temp, then upload to az://
        with tempfile.TemporaryDirectory() as td:
            tmp = LocalPath(td) / "tmp.gpkg"
            obj.to_file(tmp, driver="GPKG")
            with fsspec.open(path, "wb", **storage_options(path)) as dst, open(tmp, "rb") as src:
                shutil.copyfileobj(src, dst)

    def load_input(self, context: dg.InputContext) -> gpd.GeoDataFrame:
        path = self._get_path(context)

        with gdal_azure_session(path=path):
            return gpd.read_file(self._as_vsi(path))