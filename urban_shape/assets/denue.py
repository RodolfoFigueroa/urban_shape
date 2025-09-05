import dagster as dg
import geopandas as gpd
import pandas as pd

from upath import UPath as Path
from urban_shape.partitions import zone_partitions
from urban_shape.resources import PathResource
from cfc_core_utils import gdal_azure_session, storage_options


@dg.asset(
    name="base",
    key_prefix="denue",
    group_name="denue",
    io_manager_key="geodataframe_manager"
)
def denue(path_resource: PathResource) -> gpd.GeoDataFrame:
    fpath = Path(path_resource.jobs_path) / "denue_2023_estimaciones.parquet"
    df = pd.read_parquet(fpath, columns=("codigo_act", "longitud", "latitud"),storage_options=storage_options(fpath))
    return gpd.GeoDataFrame(df["codigo_act"], geometry=gpd.points_from_xy(df["longitud"], df["latitud"]), crs="EPSG:4326").to_crs("EPSG:6372")


@dg.asset(
    name="split",
    key_prefix="denue",
    ins={"denue_base": dg.AssetIn(["denue", "base"])},
    group_name="denue",
    partitions_def=zone_partitions,
    io_manager_key="geodataframe_manager",
)
def denue_split(context: dg.AssetExecutionContext, path_resource: PathResource, denue_base: gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    fpath = Path(path_resource.population_grids_path) / "final" / "zone_agebs" / "shaped" / "2020" / f"{context.partition_key}.gpkg"
    fpath = str(fpath).replace("az://", "/vsiaz/")

    with gdal_azure_session(path=fpath):
        df_agebs = gpd.read_file(fpath).to_crs("EPSG:6372")

    return denue_base.sjoin(df_agebs[["geometry"]], how="inner", predicate="within").drop(columns=["index_right"]).reset_index(names="index").drop_duplicates(subset=["index"]).drop(columns=["index"])