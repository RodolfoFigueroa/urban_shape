import dagster as dg
import geopandas as gpd

from hdbscan import HDBSCAN
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from urban_shape.partitions import zone_partitions
from urban_shape.resources import ClusterConfigResource


def points_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="base",
        key_prefix=[top_prefix, "points"],
        ins={"df_points": dg.AssetIn(["denue", "split"])},
        partitions_def=zone_partitions,
        required_resource_keys=set([f"{top_prefix}_config_resource"]),
        io_manager_key="geodataframe_manager",
        group_name=f"points_{top_prefix}",
    )
    def _asset(context: dg.AssetExecutionContext, df_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        config: ClusterConfigResource = getattr(context.resources, f"{top_prefix}_config_resource")
        
        xmin, ymin, xmax, ymax = df_points.to_crs("EPSG:4326").total_bounds
        crs_options = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=xmin,
                south_lat_degree=ymin,
                east_lon_degree=xmax,
                north_lat_degree=ymax,
            )
        )

        X = df_points.to_crs(crs_options[0].code).geometry.get_coordinates().to_numpy() / 100
        
        model = HDBSCAN(
            min_samples=getattr(config, "min_samples"),
            min_cluster_size=getattr(config, "min_cluster_size"),
            cluster_selection_epsilon=getattr(config, "eps"),
            core_dist_n_jobs=1
        )
        model.fit(X)

        df_points["label"] = model.labels_
        return df_points

    return _asset


def points_remove_unused_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="remove_unused",
        key_prefix=[top_prefix, "points"],
        ins={"df_points": dg.AssetIn([top_prefix, "points", "base"])},
        partitions_def=zone_partitions,
        io_manager_key="geodataframe_manager",
        group_name=f"points_{top_prefix}",
    )
    def _asset(df_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df_points = df_points.query("label != -1")
        label_counts = df_points["label"].value_counts()
        small_labels = label_counts[label_counts < 10].index.tolist()
        return df_points[~df_points["label"].isin(small_labels)]

    return _asset


dassets = [factory(top_prefix) for factory in (points_factory, points_remove_unused_factory) for top_prefix in ("coarse", "fine")]