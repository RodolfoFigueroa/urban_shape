import dagster as dg

from urban_shape import assets
from urban_shape.managers import DataFrameManager, GeoDataFrameManager
from urban_shape.resources import (
    ClusterConfigResource,
    MedoidConfigResource,
    PathResource,
)

# Resources
coarse_config_resource = ClusterConfigResource(
    eps=0,
    min_samples=5,
    min_cluster_size=930,
)

fine_config_resource = ClusterConfigResource(
    eps=0,
    min_samples=5,
    min_cluster_size=100,
)

coarse_medoid_config_resource = MedoidConfigResource(n_clusters=4)

fine_medoid_config_resource = MedoidConfigResource(n_clusters=6)

path_resource = PathResource(
    jobs_path=dg.EnvVar("JOBS_PATH"),
    data_path=dg.EnvVar("DATA_PATH"),
    population_grids_path=dg.EnvVar("POPULATION_GRIDS_PATH"),
    scian_path=dg.EnvVar("SCIAN_PATH"),
)


# Managers
dataframe_manager = DataFrameManager(
    extension=".csv",
    path_resource=path_resource,
)

geodataframe_manager = GeoDataFrameManager(
    extension=".gpkg",
    path_resource=path_resource,
)


# Definitions
defs = dg.Definitions(
    assets=dg.load_assets_from_modules(
        [assets.polygons, assets.denue, assets.points, assets.scian],
    ),
    resources={
        "fine_config_resource": fine_config_resource,
        "coarse_config_resource": coarse_config_resource,
        "fine_medoid_config_resource": fine_medoid_config_resource,
        "coarse_medoid_config_resource": coarse_medoid_config_resource,
        "path_resource": path_resource,
        "dataframe_manager": dataframe_manager,
        "geodataframe_manager": geodataframe_manager,
    },
)
