from pathlib import Path

import dagster as dg
import geopandas as gpd
import pandas as pd

from urban_shape.resources import PathResource


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


class DataFrameManager(BaseManager):
    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        path = self._get_path(context)
        return pd.read_csv(path)


class GeoDataFrameManager(BaseManager):
    def handle_output(self, context: dg.OutputContext, obj: gpd.GeoDataFrame) -> None:
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_file(path)

    def load_input(self, context: dg.InputContext) -> gpd.GeoDataFrame:
        path = self._get_path(context)
        return gpd.read_file(path)
